#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout + 1w EMA(21) trend + volume confirmation
    # Long when: price breaks above Camarilla H3 (1d) AND price > 1w EMA21 AND volume > 2.0x 20-bar avg volume
    # Short when: price breaks below Camarilla L3 (1d) AND price < 1w EMA21 AND volume > 2.0x 20-bar avg volume
    # Exit when: price crosses Camarilla pivot point (PP) OR adverse 1w EMA21 crossover
    # Uses discrete sizing (0.25) targeting 30-100 total trades over 4 years.
    # Camarilla levels provide precise support/resistance; 1w EMA21 filters counter-trend moves on weekly trend;
    # Volume spike confirms breakout validity. Works in bull (trend continuation) and bear (mean-reversion at PP).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    PP_1d = (high_1d + low_1d + close_1d) / 3.0
    H3_1d = PP_1d + (high_1d - low_1d) * 1.1 / 4.0
    L3_1d = PP_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Get 1w data for EMA(21) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed bars)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(PP_1d_aligned[i]) or np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > H3_1d_aligned[i-1]  # break above previous H3
        breakout_down = close[i] < L3_1d_aligned[i-1]  # break below previous L3
        
        # 1w EMA21 trend filter
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < PP_1d_aligned[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > PP_1d_aligned[i] or not downtrend))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_ema21_volume_v1"
timeframe = "1d"
leverage = 1.0