#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# Williams %R identifies overbought/oversold conditions on 6h timeframe
# 1d EMA50 filter ensures we only trade in direction of higher timeframe trend
# Volume spike (>2.0x 20-period average) confirms institutional participation
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Works in both bull and bear markets by combining mean reversion with trend filter

name = "6h_WilliamsR_MeanRev_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    start_idx = max(14, 50, 20)  # Williams %R, EMA50, and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Long entry: Williams %R oversold (< -80) + price above 1d EMA50 + volume confirmation
        if curr_williams_r < -80.0 and curr_close > curr_ema_50_1d and vol_confirm:
            signals[i] = 0.25
        # Short entry: Williams %R overbought (> -20) + price below 1d EMA50 + volume confirmation
        elif curr_williams_r > -20.0 and curr_close < curr_ema_50_1d and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals