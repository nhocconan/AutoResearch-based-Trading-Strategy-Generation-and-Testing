#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Camarilla_Breakout_v3
Hypothesis: On daily timeframe, Camarilla R3/S3 breakouts aligned with weekly EMA34 trend and volume confirmation capture major trend moves while avoiding false breakouts in chop. Weekly trend filter ensures we only trade with the dominant higher-timeframe momentum. Volume spike confirms institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag and maximize edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data ONCE before loop for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (R3, S3) using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous day close
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d_shifted + 1.1 * camarilla_range / 4
    s3 = close_1d_shifted - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to daily timeframe (no alignment needed as primary TF is 1d)
    r3_aligned = r3  # already at 1d frequency
    s3_aligned = s3
    
    # Daily volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for weekly EMA and volume MA)
    start_idx = max(35, 20)  # weekly EMA34 needs 34 bars + 1 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter (EMA34)
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Long logic: breakout above R3 in uptrend with volume
        if uptrend and volume_spike and breakout_r3:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S3 in downtrend with volume
        elif downtrend and volume_spike and breakout_s3:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_Camarilla_Breakout_v3"
timeframe = "1d"
leverage = 1.0