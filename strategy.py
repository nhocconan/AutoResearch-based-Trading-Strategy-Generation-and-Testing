#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves in both bull and bear markets
# 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>1.5x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) balances return potential with fee minimization
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) from 1d data
    # We need to calculate these on 1d data then align to 1d timeframe (identity)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1w EMA50 and Donchian/volume warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_donchian_high = high_roll[i]
        curr_donchian_low = low_roll[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        vol_spike = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR breaks 1w EMA50 trend
            if curr_close < curr_donchian_low or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR breaks 1w EMA50 trend
            if curr_close > curr_donchian_high or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND above 1w EMA50 AND volume spike
            if curr_high > curr_donchian_high and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below 1w EMA50 AND volume spike
            elif curr_low < curr_donchian_low and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals