# -*- mode: ruby -*-
# vi: set ft=ruby :

VAGRANTFILE_API_VERSION = "2"
VM_RAM = 8
VM_CPUS = 6

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|

  config.vm.box = "bento/ubuntu-16.04"

  config.vm.network "forwarded_port", guest: 8888, host: 8888

  config.vm.synced_folder ".", "/vagrant"
  
  config.vm.provider "virtualbox" do |v|
    v.memory = 1024 * VM_RAM
    v.cpus = VM_CPUS
  end

  config.vm.provision "shell", path: "provision.sh"

end
