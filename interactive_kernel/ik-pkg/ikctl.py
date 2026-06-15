"""Control a running interactive kernel from a terminal (out-of-band).

Usage:
    python ikctl.py status
    python ikctl.py pause [timeout]
    python ikctl.py stack
    python ikctl.py locals [depth]
    python ikctl.py set NAME EXPR [DEPTH]      e.g.  set lr 1e-4
    python ikctl.py resume | stop

Connects to the most recently started kernel, or pass --conn <file>.
"""
import sys
from jupyter_client import BlockingKernelClient, find_connection_file


def main(argv):
    conn = None
    if "--conn" in argv:
        i = argv.index("--conn"); conn = argv[i + 1]; del argv[i:i + 2]
    cmd, args = argv[0], argv[1:]
    client = BlockingKernelClient(connection_file=conn or find_connection_file())
    client.load_connection_file()
    client.start_channels()
    try:
        if cmd == "pause":
            content = {"timeout": float(args[0]) if args else 5.0}
            reply = _control(client, "ik_pause_request", content)
        elif cmd == "resume":
            reply = _control(client, "ik_resume_request", {})
        elif cmd == "stop":
            reply = _control(client, "ik_stop_request", {})
        elif cmd == "status":
            reply = _control(client, "ik_status_request", {})
        elif cmd == "stack":
            reply = _control(client, "ik_inspect_request", {})
            for f in reply["content"].get("stack", []):
                print(f"[{f['depth']}] {f['function']}()  {f['file']}:{f['line']}")
            return
        elif cmd == "locals":
            depth = int(args[0]) if args else 0
            reply = _control(client, "ik_inspect_request", {"depth": depth})
            for k, v in reply["content"].get("locals", {}).items():
                print(f"  {k} = {v}")
            return
        elif cmd == "set":
            content = {"name": args[0], "value_expr": args[1],
                       "depth": int(args[2]) if len(args) > 2 else 0}
            reply = _control(client, "ik_set_variable_request", content)
        else:
            print(__doc__); return
        print(reply["content"])
    finally:
        client.stop_channels()


def _control(client, msg_type, content, timeout=15):
    msg = client.session.msg(msg_type, content)
    client.control_channel.send(msg)
    while True:
        reply = client.control_channel.get_msg(timeout=timeout)
        if reply["parent_header"].get("msg_id") == msg["header"]["msg_id"]:
            return reply


if __name__ == "__main__":
    main(sys.argv[1:] or ["status"])
