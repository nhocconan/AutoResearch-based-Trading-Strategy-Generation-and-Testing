#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data once for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 6
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 6
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume average (20-period) for volume spike detection
    vol_ma20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike AND above daily EMA34
            long_cond = (close[i] > R3_aligned[i]) and (volume[i] > 1.5 * vol_ma20_aligned[i]) and (close[i] > ema34_aligned[i])
            
            # Short: Price breaks below S3 with volume spike AND below daily EMA34
            short_cond = (close[i] < S3_aligned[i]) and (volume[i] > 1.5 * vol_ma20_aligned[i]) and (close[i] < ema34_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses back below R3
            if close[i] < R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses back above S3
            if close[i] > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance.
# Breakouts with volume confirmation and aligned with daily trend (EMA34) capture momentum.
# Long when price breaks above R3 with volume spike and above daily EMA34.
# Short when price breaks below S3 with volume spike and below daily EMA34.
# Exits when price returns inside the level.
# Uses daily timeframe for Camarilla calculation and trend filter to avoid look-ahead.
# Volume spike filter (1.5x 20-day average) ensures commitment.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.