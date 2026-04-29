#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND 1d EMA34 uptrend AND volume spike
# Short when price breaks below Donchian(20) low AND 1d EMA34 downtrend AND volume spike
# Exit when price returns to Donchian(20) midpoint OR trend reverses
# Donchian channels provide clear structure, 1d EMA34 filters higher timeframe trend,
# volume confirmation ensures momentum validity
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, donchian_window)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price below Donchian midpoint OR 1d EMA34 downtrend
            if curr_close < donchian_mid[i] or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Donchian midpoint OR 1d EMA34 uptrend
            if curr_close > donchian_mid[i] or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price above 1d EMA34 AND volume spike
            if (curr_high > donchian_high[i] and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price below 1d EMA34 AND volume spike
            elif (curr_low < donchian_low[i] and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals