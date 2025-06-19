%% Assessing the effects of coupling
function Figure4()

    tspan = [0 120000]; % Total simulation time in ms
    
    N = 10; % Number of cells

    % Initial conditions (Fletcher et al. 2017)
    v0 = -61.28;    % Initial membrane voltage (mV)
    c0 = 0.078;     % Initial cytosolic calcium (μM)
    hna0 = 0.707;  % INa inactivation gate
    n0 = 0.0;      % IK activation gate
    nbk0 = 0.0;    % IBK activation gate
    ha0 = 0.707;   % IA inactivation gate
    ATP0 = 0.01;   % Initial ATP (uM)
    c_er0 = 83.16; % Initial calcium in ER (uM) 
    IP30 = 0;      % Initial IP3
    h0 = 0.889;    % Initial Inactivation variable h = Kd/ (c + Kd): steady state value

    V_fs0 = -61; % Initial FS Cell membrane voltage, mV
    w_fs0 = 0.0148; % Initial recovery variable value

    y0_single = [v0; c0; hna0; n0; nbk0; ha0; ATP0; c_er0; IP30; h0; V_fs0; w_fs0];
    y0 = repmat(y0_single, N, 1);

    n_conditions = 3;

    for condition = 1:n_conditions
    
        if condition == 1 % No coupling

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 0;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 2 % Gap Junction

            conditions(condition).ggj = 0.05;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 0;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 3 % Calcium Diffusion

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0.05;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 0;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 4 % Gap Junction and Calcium Diffusiom

            conditions(condition).ggj = 0.05;
            conditions(condition).Dca = 0.05;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 0;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 5 % ATP - with ATP diffusion

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 0.05; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 6 % ATP (acting through P2Y) and IP3 - with IP3 diffusion (no gp2x, essentially just IP3)

            conditions(condition).ggj = 0.;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 1; conditions(condition).dIP3_switch = 1; conditions(condition).IP3diff = 1;
            conditions(condition).FS_switch = 0;

        elseif condition == 7 % ATP (acting through P2Y) and IP3 - no IP3 diffusion (no gp2x, essentially just IP3)

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 1; conditions(condition).dIP3_switch = 1; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 8 % ATP (acting through P2X and P2Y) and IP3 - including ATP and IP3 diffusion

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 1; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 1; conditions(condition).dIP3_switch = 1; conditions(condition).IP3diff = 1;
            conditions(condition).FS_switch = 0;

        elseif condition == 9 % ATP (acting through P2X and P2Y) and IP3 - including ATP diffusion and no IP3 diffusion

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 1; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 1; conditions(condition).dIP3_switch = 1; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 0;

        elseif condition == 10 % All coupling without FS cells

            conditions(condition).ggj = 0.05;
            conditions(condition).Dca = 0.05;
            conditions(condition).gp2x = 1; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 1; conditions(condition).dIP3_switch = 1; conditions(condition).IP3diff = 1;
            conditions(condition).FS_switch = 0;

        elseif condition == 11 % FS cells, no gap junctions between corticotrophs

            conditions(condition).ggj = 0;
            conditions(condition).Dca = 0;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 0;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 1;
        
        elseif condition == 12 % FS cells with gap junctions between corticotrophs
        
            conditions(condition).ggj = 0.05;
            conditions(condition).Dca = 0.05;
            conditions(condition).gp2x = 0; conditions(condition).dATP_switch = 0;
            conditions(condition).Jip3_switch = 0; conditions(condition).dIP3_switch = 0; conditions(condition).IP3diff = 0;
            conditions(condition).FS_switch = 1;

        elseif condition == 13 % All coupling and FS cells

            conditions(condition).ggj = 0.05;
            conditions(condition).Dca = 0.05;
            conditions(condition).gp2x = 1; conditions(condition).dATP_switch = 1;
            conditions(condition).Jip3_switch = 1; conditions(condition).dIP3_switch = 1; conditions(condition).IP3diff = 1;
            conditions(condition).FS_switch = 1;

        end
    
    end
    
    n_repeats = 10;

    corr_average = zeros(n_repeats,n_conditions);
   
    load('all_params_full_CRH.mat');

    for condition = 1:n_conditions
        disp(condition);

        for rep = 1:n_repeats
            
                
            params = all_params((rep-1)*N + 1 : rep*N); % extract parameters for this repeat
    
            [t, Y] = ode15s(@(t, y) coupled_model(t, y, params, condition, conditions, N, tspan), tspan, y0); % solve differential equations
            
            C_traces = zeros(60000,N);
    
            for j = 1:N
    
                idx = (j-1)*12;
                c = Y(:,idx+2);
    
                C_traces(:,j) = interp1(t, c, linspace(tspan(1), tspan(end), 60000));
    
            end
    
            all_corrs = corr_calc(C_traces);
    
            corr_average(rep,condition) = mean(all_corrs(4:end)); % 2nd frame to end to remove intialisation frame(1)
            % 4th frame to end for CRH: frame 1 = initialisation, frame 2 = spon activity, frame 3 = CRH response, frame 4 to end = post CRH
            % disp(corr_average(rep,condition));
    
        end

        tot_averages(condition) = mean(corr_average(:,condition));

    end

    save('all_params_full_CRH.mat','all_params');
    save('corr_average_FS_3.mat','corr_average');
    save('tot_averages_FS_3.mat','tot_averages');

    figure;
    bar(tot_averages);

end

%% Correlation Analysis
function all_windows = corr_calc(C_traces)


    winSize = 7500;
    stepSize = 7500;
    threshold = 0.75;
    
    [T, node] = size(C_traces);
    
    % Window loop
    frameCount = 0;
    all_windows = zeros(1,(T/stepSize));
    startIndices = 1:stepSize:(T - winSize +1);
    if startIndices(end) < (T - winSize + 1)
        startIndices = [startIndices, T - winSize + 1];
    end
    for sIdx = 1:length(startIndices)
        startIdx = startIndices(sIdx);
        frameCount = frameCount + 1;
        endIdx = startIdx + winSize - 1;
        windowData = C_traces(startIdx:endIdx, :);
        % Compute Pearson correlation matrix
        R = corrcoef(windowData);
    
        % Binarize for functional connectivity
        adj = (R > threshold) & ~eye(node);

        % Calculate percentage correlation
        percentage_correlation = ((sum(adj(:)==1)/2)/45)*100;
        
        all_windows(1, frameCount) = percentage_correlation; % returns the percentage correlation for each window
        
    end

end

%% Model equations
function dydt = coupled_model(t, y, params, condition, conditions, N, tspan)
    
    dydt = zeros(size(y));

    
    for i = 1:N
        idx = (i-1)*12 + 1; % Currently 12 variables
    
        % State Variables
        v = y(idx);      % Membrane potential (mV)
        c = y(idx+1);      % Cytosolic calcium (μM)
        hna = y(idx+2);    % INa inactivation variable
        n = y(idx+3);      % IK activation variable
        nbk = y(idx+4);    % IBK activation variable
        ha = y(idx+5);     % IA inactivation variable
        ATP = y(idx+6);    % ATP
        c_er = y(idx+7);   % ER Calcium
        IP3 = y(idx+8);    % IP3
        h = y(idx+9);      % Inactivation variable
        V_fs = y(idx+10); % FS cell membrane potential (mV)
        w_fs = y(idx+11); % FS cell recovery variable

        %% Pulse Parameters
        tau = 12500;            % Time constant of pulse (ms)
        tpulse = 30000;    % Time pulse starts (ms) = tspan(2) [for spon activity] | = 30000 [for CRH stimulation]
        delpulse = t - tpulse;  % Time after pulse starts
        kp = 1;                 % Pulse half-saturation constant
        pnorm = 4 * tau^2 * exp(-2);  % Normalizing constant
        pulse = (delpulse.^2 ./ pnorm) .* exp(-delpulse ./ tau) .* (delpulse > 0);  % Pulse profile
        g = (pulse ./ (pulse + kp)) * (1 + kp);  % Pulse effect scaling variable

        %% Reversal Potentials (mV)
        Ek = -75;     % K+ reversal
        Eca = 60;     % Ca2+ reversal
        Ena = 75;     % Na+ reversal
        Ens = 0;      % Non-specific reversal
        ECa_fs = 120; % FS cell Ca2+ reversal
        EK_fs = -74;  % FS cell K+ reversal
        Eleak_fs = -60; % FS cell leak reversal
    
        %% Driving forces
        vkdrive = v - Ek;     % Driving force for K+
        vnadrive = v - Ena;   % Driving force for Na+
        vcadrive = v - Eca;   % Driving force for Ca2+
        vnsdrive = v - Ens;   % Driving force for non-specific
    
        %% Membrane capacitance
        Cm = 6.0;  % (pF)
        Cm_fs = 20; % FS cell
    
        %% Passive Parameters

        gkleak = 0.1;    % K+ leak conductance, (default = 0.1)
        gkleakp = 0;     % K+ leak pulse component
        gkirpulse = 0;   % Kir pulse component
        gCRHpulse = 0;   % CRH pulse conductance
        kcabk = 1.5;     % BK Ca2+ sensitivity
        CRHkcabk = 0;    % Pulse modification of kcabk
        vmca = -12;      % ICa half-activation voltage
        CRHvmca = 0;     % CRH modulation of vmca

        if t < 30000 % = tspan(2) [for spon activity] | = 30000 [for CRH stimulation]
            gCRH = 0;
            gkir = 1.3;
        else
            gCRH = params(i).delta_gCRH; % Increase gCRH
            gkir = 1.3 - params(i).delta_gkir; % Decrease gkir
        end
          
        %% INa (TTX-sensitive fast Na+ current)
        gnav = 10;             % Max conductance
        tauhna = 2;            % Time constant for inactivation
        vmna = -15; smna = 5;  % Activation parameters
        vhna = -60; shna = -10;% Inactivation parameters
        mnainf = 1 / (1 + exp((vmna - v)/smna));  % Steady-state activation
        hnainf = 1 / (1 + exp((vhna - v)/shna));  % Steady-state inactivation
        inav = gnav * mnainf^3 * hna * vnadrive;  % Current
    
        %% ICa (L-type Ca2+ current)
        gca = 1.4; smca = 10;
        mcainf = 1 / (1 + exp(((vmca + CRHvmca * g) - v)/smca));  % Activation
        ical = gca * mcainf * vcadrive;  % L-type Ca2+ current
    
        %% ICa,leak
        gcaleak = 0.025; % Small constant leak
        icaleak = gcaleak * vcadrive;
        ica = ical + icaleak;  % Total Ca2+ current
    
        %% IK (delayed rectifier K+ current)
        gk = 2.2; vn = 0; sn = 5; taun = 20;
        ninf = 1 / (1 + exp((vn - v)/sn));  % Activation
        ikdr = gk * n * vkdrive;
    
        %% IA (A-type transient K+ current)
        ga = 0.5;
        tauha = 20;
        vna = -20; sna = 10; vha = -60; sha = -10;
        nainf = 1 / (1 + exp((vna - v)/sna));  % Activation
        hainf = 1 / (1 + exp((vha - v)/sha));  % Inactivation
        ia = ga * nainf * ha * vkdrive;
    
        %% IKCa (small conductance, far from Ca2+ source)
        gkca = 2; kkca = 0.4;
        c2 = c^2;
        nkcainf = c2 / (c2 + kkca^2);  % Activation from cytosolic Ca2+
        ikca = gkca * nkcainf * vkdrive;
    
        %% IBK (large conductance, near Ca2+ source)
        taunbk = 5; vbk0 = 0.1; snbk = 5;
        A = 0.11; kshift = 18;
        cad = -A * ical;  % Domain calcium from L-type Ca2+
        vnbk = vbk0 - kshift * log(cad / (kcabk + CRHkcabk * g));  % V0(BK)
        nbkinf = 1 / (1 + exp((vnbk - v)/snbk));  % Activation
        ibk = params(i).gbk * nbk * vkdrive;
    
        %% IKir (inward rectifier K+)
        vnkir = -55; snkir = -10;
        nkirinf = 1 / (1 + exp((vnkir - v)/snkir));
        ikir = (gkir + gkirpulse * g) * nkirinf * vkdrive;
    
        %% IKleak (leak K+ current)
        ikleak = (gkleak + gkleakp * g) * vkdrive;
    
        %% Total K+ current
        ik = ikdr + ikca + ibk + ikir + ia + ikleak;
    
        %% INab (non-specific background Na+)
        inab = params(i).gnab * vnsdrive;
    
        %% ICRH (non-specific background CRH-sensitive current)
        iCRH = (gCRH + gCRHpulse * g) * vnsdrive;
    
        %% Total non-specific current
        ins = inab + iCRH;
        
        %% Gap junction coupling
        left = mod(i-2, N)+1; % Index cell to the left
        right = mod(i, N)+1;  % Index cell to the right
        v_left = y((left-1)*12 + 1);
        v_right = y((right-1)*12 +1);
        igj = conditions(condition).ggj*((v_left + v_right) - 2*v);

        %% Calcium handling
        fcyt = 0.01;     % Cytosol fraction
        alpha = 0.0015;  % Ca2+ influx conversion factor
        Jin = -alpha * ica;  % Membrane Ca2+ influx
        nuc = 0.03; kc = 0.1;
        Jout = nuc * c2 / (c2 + kc^2);  % Ca2+ efflux via pump
        Jmem = Jin - Jout;  % Net flux across membrane
        c_left = y((left-1)*12 + 2); % Calcium in the left neighbour
        c_right = y((right-1)*12 + 2); % Calcium in the right neighbour

        % Calcium diffusion
        dcdiff = conditions(condition).Dca* (c_left + c_right - 2*c);
    
        %% ATP Coupling
        ATP_left = y((left-1)*12 + 7); % ATP in the left neighbour
        ATP_right = y((right-1)*12 + 7); % ATP in the right neighbour
        ATP_release = 0.15; % calcium threshold required for ATP release (uM)
        k_ATP = 0.05; % ATP release rate (uM/ms)
        k_deg = 0.005; % ATP degradation rate (ms-1)
        Kd_ATP = 5; % ATP half-activation (uM)
        D_ATP = 0.05; % ATP diffusion coefficeint                   
        ip2x = conditions(condition).gp2x * (ATP^2/(Kd_ATP + ATP^2)) * vnsdrive;
        Jp2x = alpha * ip2x;
                         
        %% IP3 (Parameters from Function 2022 Mammano)
        L = 0.0001; % leak (ms^-1)
        Pip3r = 42.625; % maximum permeability of IP3 channels (uM/ms)
        Ki = 1; % dissociation constant of IP3 on IP3R (uM)
        Ka = 0.4; % dissociation constant of calcium activation on IP3R (uM)
        Vserca = 0.04; % non-scaled maximal SERCA pump rate (uM/ms) Calculated to match the constant pump (0.1*c from Fletcher)
        Kserca = 0.2; % Ca2+ level for half maximal activation of the SERCA pump (uM)
        if conditions(condition).Jip3_switch == 1
            Jip3 = (L + (Pip3r * IP3^3 * c^3 * h^3) / ((IP3 + Ki)^3 * (c + Ka)^3)) * (c_er - c); % Calcium flux due to IP3
            Jserca = (Vserca * c^2) / (c^2 + Kserca^2); % Calcium flux due to SERCA pump         
        else
            Jip3 = 0;
            Jserca = 0;
        end
        A = 0.005; % controls the time scale between differential eqns (uM^-1ms^-1)
        Kd = 0.4; % dissocation constant of calcium inactivation on IP3R (uM)        
        V_PLC = 0.00185; % maximal rate of PLC production (uM/ms)
        K_PLC = 1.03; % ATP EC50 value for PLC production (uM)
        k_IP3deg = 0.015; % IP3 degradation rate (ms^-1)
        D_IP3 = 0.05; % IP3 diffusion constant (um^2/ms) 
        IP3_left = y((left-1)*12 + 9); % IP3 in left neighbour
        IP3_right = y((right-1)*12 + 9); % IP3 in right neighbour
        fer = 0.01; % fraction of free calcium in ER
        sigmav = 31; % ratio of cytosolic to ER volume
        Jer = Jip3 - Jserca;

        %% FS cell
        i_app = 0; % changes the cell from silent to spiking (>90)     
        gL_fs = 2; % nS        
        iL_fs = gL_fs * (V_fs - Eleak_fs);     
        gCa_fs = 4.4; % Calcium channel conductance, nS (1.4)
        V1_fs = -1.2; % Half-activation voltage, mV
        V2_fs = 18; % Slope factor, mV
        M_inf_fs = 0.5 * (1 + tanh((V_fs - V1_fs)/V2_fs));
        iCa_fs = gCa_fs * M_inf_fs * (V_fs - ECa_fs);        
        gK_fs = 8; % Potassium channel conductance, nS (2.2)
        V3_fs = 2; % Half-activation voltage, mV
        V4_fs = 30; % Slope factor, mV
        psi_fs = 0.04; % Rate constant scaling, ms^-1        
        w_inf_fs = 0.5 * (1 + tanh((V_fs - V3_fs)/V4_fs));
        tau_w_fs = 1 / (psi_fs * cosh((V_fs - V3_fs)/(2*V4_fs)));        
        iK_fs = gK_fs * w_fs * (V_fs - EK_fs);
        V_fs_left = y((left-1)*12 + 11);
        V_fs_right = y((right-1)*12 + 11);
        ggj_fs = 1;
        igj_fsfs = conditions(condition).FS_switch * (ggj_fs * ((V_fs_left + V_fs_right) - 2*V_fs));
        ggj_cc = 0.01;
        igj_fscc = conditions(condition).FS_switch * (ggj_cc * (v - V_fs)); % current into FS cell
        igj_ccfs = conditions(condition).FS_switch * (ggj_cc * (V_fs - v)); % current into CC

        %% Differential equations
        dv = -(inav + ica + ik + ins - igj + ip2x + igj_ccfs) / Cm;    % Membrane potential
        dc = fcyt * (Jmem + dcdiff + Jp2x + Jer); % Calcium concentration
        
        dhna = (hnainf - hna) / tauhna;        % INa inactivation
        dn = (ninf - n) / taun;                % IK activation
        dnbk = (nbkinf - nbk) / taunbk;        % IBK activation
        dha = (hainf - ha) / tauha;            % IA inactivation
        
        % Switch = 0 (dATP = 0) of 1 (dATP = ...)
        dATP = conditions(condition).dATP_switch * (k_ATP * (c > ATP_release) - k_deg * ATP + D_ATP * (ATP_left + ATP_right - 2*ATP)) ; % ATP
       
        dc_er = -fer * sigmav * Jer; % Calcium in ER

        % Switch = 0 (dIP3 = 0) of 1 (dIP3 = ...), IP3diff = 0 (no diffusion) or 1 (diffusion)
        dIP3 = conditions(condition).dIP3_switch * (V_PLC * (ATP / (K_PLC + ATP)) - k_IP3deg * IP3 + conditions(condition).IP3diff * D_IP3*(IP3_left + IP3_right - 2*IP3)); % IP3      
        dh = A * (Kd - (c + Kd) * h); % IP3 Inactivation variable

        dV_fs = conditions(condition).FS_switch * ((i_app - iL_fs - iCa_fs - iK_fs + igj_fscc + igj_fsfs) / Cm_fs); % FS cell membrane potential
        dw_fs = conditions(condition).FS_switch * ((w_inf_fs - w_fs) / tau_w_fs); % FS cell recovery variable

        dydt(idx:idx+11) = [dv; dc; dhna; dn; dnbk; dha; dATP; dc_er; dIP3; dh; dV_fs; dw_fs];
        
    end
end

