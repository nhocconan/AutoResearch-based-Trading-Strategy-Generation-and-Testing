#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMA smoothed 8), Teeth (8-period SMA smoothed 5), Lips (5-period SMA smoothed 3)
# Long: Lips > Teeth > Jaw AND price above 1d EMA34 AND volume spike
# Short: Lips < Teeth < Jaw AND price below 1d EMA34 AND volume spike
# Works in trending markets (alligator eating) and avoids whipsaws in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    
    # Williams Alligator components (using 12h data)
    # Jaw: 13-period SMA, then smoothed by 8-period SMA
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(sma_13).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, then smoothed by 5-period SMA
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(sma_8).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, then smoothed by 3-period SMA
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(sma_5).rolling(window=3, min_periods=3).mean().values
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 8, 5, 20)  # 1d EMA34, Alligator components, and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator sleeping (Lips < Teeth OR Teeth < Jaw) OR price breaks below 1d EMA34
            if curr_lips < curr_teeth or curr_teeth < curr_jaw or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator sleeping (Lips > Teeth OR Teeth > Jaw) OR price breaks above 1d EMA34
            if curr_lips > curr_teeth or curr_teeth > curr_jaw or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (Alligator eating up) AND price above 1d EMA34 AND volume spike
            if curr_lips > curr_teeth and curr_teeth > curr_jaw and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (Alligator eating down) AND price below 1d EMA34 AND volume spike
            elif curr_lips < curr_teeth and curr_teeth < curr_jaw and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals