#!/usr/bin/env bash

echo

if [[ $(uname -m) == 'arm64' ]]; then

    machine=$(podman machine list -q)

    # We need a virtual machine for macOS
    if [ ! $machine ]; then
        echo "Podman does not have a virtual machine. Want to create one?"
        
        if gum confirm "Create machine?" --default=true --affirmative "Yes" --negative "No"; then
            gum spin --spinner dot --title "Initializing machine..." -- podman machine init 2>&1
            machine=$(podman machine list -q)
            echo "✅ Podman virtual machine created. It can be removed with 'podman machine rm'."
            echo
        else
            echo "Okay. You can create one later with 'podman machine init'."
            echo
        fi
    fi

    autostart="$HOME/.config/podman-env/autostart"

    # If we have a machine now it's because we *want* one
    if [ $machine ]; then
    	if [[ "$( podman machine inspect --format '{{.State}}' )" == "running" ]]; then
            echo "✅ Podman virtual machine is running."
        else
            if [ -f $autostart ]; then
                startmachine=1
            else
                startmachine=0

                echo "Would you like to start the Podman virtual machine?"
                choice=$(gum choose "Always - start now & on future activations" "Yes - start now only" "No - do not start")

                case ${choice:0:1} in
                    'A')
                        echo "> Always"
                        mkdir -p $HOME/.config/podman-env
                        echo "1" > $autostart
                        echo
                        echo "Machine will start automatically on next activation. To disable this, run:"
                        echo "  rm $autostart"
                        echo
                        startmachine=1
                        ;;
                    'Y')
                        echo "> Yes"
                        startmachine=1
                        ;;
                    'N')
                        echo "> No"
                        echo; echo "You can start the machine with 'podman machine start'"
                        startmachine=0
                        ;;
                    *)
                        echo "> Unknown response"
                        ;;
                esac
            fi

            if [ $startmachine = 1 ]; then
                gum spin --spinner dot --title "Starting machine..." -- podman machine start 2>&1
                echo "✅ Podman virtual machine started - stop it with 'podman machine stop'."
            fi
        fi
    fi
fi

echo "🍟 Podman is now available."