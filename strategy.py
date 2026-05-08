#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour 4-hour Bollinger Band breakout with daily trend filter and volume confirmation
# We go long when price breaks above upper BB(20,2) on 4h with daily EMA(50) uptrend and volume spike on 1h.
# We go short when price breaks below lower BB(20,2) on 4h with daily EMA(50) downtrend and volume spike on 1h.
# Uses 1h timeframe for precise entry timing, 4h for signal direction, and daily for trend filter.
# Bollinger Bands capture volatility-based support/resistance levels.
# Daily trend filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.
# Designed to generate 15-37 trades per year on 1h timeframe to avoid fee drag.

name = "1h_BollingerBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 4h close
    close_4h = df_4h['close'].values
    bb_ma = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Align Bollinger Bands to 1h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current 1h volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + daily uptrend + volume spike
            if (not np.isnan(bb_upper_val) and close[i] > bb_upper_val and 
                close[i] > ema50_1d_val and vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below lower BB + daily downtrend + volume spike
            elif (not np.isnan(bb_lower_val) and close[i] < bb_lower_val and 
                  close[i] < ema50_1d_val and vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower BB OR daily trend turns down
            if (not np.isnan(bb_lower_val) and close[i] < bb_lower_val) or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above upper BB OR daily trend turns up
            if (not np.isnan(bb_upper_val) and close[i] > bb_upper_val) or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals