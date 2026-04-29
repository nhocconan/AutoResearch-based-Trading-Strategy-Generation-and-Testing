#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakout captures strong momentum moves, EMA34 ensures alignment with higher timeframe trend
# Volume spike (>1.8x 30-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn while maintaining meaningful exposure
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 30-period average volume for spike confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 30)  # 1d EMA34, Donchian(20), volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_vol_ma = vol_ma_30[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.8x 30-period average
        vol_spike = curr_volume > 1.8 * curr_vol_ma
        
        # Donchian breakout conditions
        breakout_up = curr_high > curr_high_20
        breakout_down = curr_low < curr_low_20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR breaks 1d EMA34 trend
            if curr_close < curr_low_20 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR breaks 1d EMA34 trend
            if curr_close > curr_high_20 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish Donchian breakout AND above 1d EMA34 AND volume spike
            if breakout_up and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Donchian breakout AND below 1d EMA34 AND volume spike
            elif breakout_down and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals