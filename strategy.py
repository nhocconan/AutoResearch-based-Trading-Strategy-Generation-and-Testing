#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) zero-cross with 1d EMA34 trend filter and volume spike confirmation
# TRIX(9) captures medium-term momentum with smoothing to reduce whipsaw
# EMA34 on 1d ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period average) confirms institutional participation and reduces false breakouts
# Discrete position sizing (0.25) balances return potential with fee minimization
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_TRIX9_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate TRIX(9): triple EMA of close, then ROC
    # TRIX = 100 * (EMA3(EMA2(EMA1(close))) - prev) / prev
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix_values = trix.values
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 27)  # EMA34(1d) + volume MA + TRIX warmup (3*9)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_trix = trix_values[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_spike = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR breaks 1d EMA34 trend
            if curr_trix < 0 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR breaks 1d EMA34 trend
            if curr_trix > 0 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: TRIX crosses above zero AND above 1d EMA34 AND volume spike
            if curr_trix > 0 and trix_values[i-1] <= 0 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero AND below 1d EMA34 AND volume spike
            elif curr_trix < 0 and trix_values[i-1] >= 0 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals