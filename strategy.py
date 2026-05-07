#!/usr/bin/env python3
name = "4h_Donchian20_1dTrend_VolumeSqueeze_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume squeeze: volume below 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + daily uptrend + volume squeeze
            vol_squeeze = volume[i] < vol_ma_20[i] * 0.8
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > high_20[i] and vol_squeeze and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + daily downtrend + volume squeeze
            elif close[i] < low_20[i] and vol_squeeze and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price retrace to midline or trend reversal
            mid = (high_20[i] + low_20[i]) / 2
            if close[i] < mid or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price retrace to midline or trend reversal
            mid = (high_20[i] + low_20[i]) / 2
            if close[i] > mid or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout with daily trend filter and volume squeeze
# - Donchian breakout captures momentum after consolidation
# - Volume squeeze (low volume) indicates compressed volatility before breakout
# - Daily EMA(50) ensures trading with higher timeframe trend
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit at midline or trend reversal to capture swing moves
# - Position size 0.25 limits risk and keeps trade frequency ~20-40/year
# - Avoids overtrading by requiring volume squeeze + trend alignment
# - Effective in both trending and ranging markets due to squeeze filter