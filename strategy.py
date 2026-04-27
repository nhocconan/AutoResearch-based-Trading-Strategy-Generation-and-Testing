#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 34-period EMA with 1-week Supertrend trend filter and volume confirmation.
# Long when price closes above EMA34 and weekly Supertrend uptrend, short when price closes below EMA34 and weekly Supertrend downtrend.
# EMA34 provides smooth trend direction, weekly Supertrend ensures multi-timeframe alignment.
# Volume confirmation (>1.5x 20-period average) filters low-conviction moves.
# Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drift.
# Works in bull markets (captures sustained uptrends) and bear markets (captures sustained downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR for Supertrend (period=10) on weekly data
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation on weekly data
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    # Initialize Supertrend arrays
    supertrend_1w = np.full(len(close_1w), np.nan)
    direction_1w = np.full(len(close_1w), 1)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_idx = 10
    if len(close_1w) > start_idx:
        supertrend_1w[start_idx] = upper_band_1w[start_idx]
        direction_1w[start_idx] = 1
    
    for i in range(start_idx + 1, len(close_1w)):
        if np.isnan(atr_1w[i]) or np.isnan(upper_band_1w[i]) or np.isnan(lower_band_1w[i]):
            supertrend_1w[i] = supertrend_1w[i-1]
            direction_1w[i] = direction_1w[i-1]
            continue
            
        if close_1w[i] <= supertrend_1w[i-1]:
            direction_1w[i] = -1
        else:
            direction_1w[i] = 1
            
        if direction_1w[i] == 1:
            supertrend_1w[i] = max(lower_band_1w[i], supertrend_1w[i-1])
        else:
            supertrend_1w[i] = min(upper_band_1w[i], supertrend_1w[i-1])
    
    # Align weekly Supertrend to daily timeframe
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(supertrend_1w_aligned[i]) or 
            np.isnan(direction_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above EMA34, weekly Supertrend uptrend, volume
        if (close[i] > ema34_1d_aligned[i] and 
            direction_1w_aligned[i] == 1 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below EMA34, weekly Supertrend downtrend, volume
        elif (close[i] < ema34_1d_aligned[i] and 
              direction_1w_aligned[i] == -1 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal on either timeframe
        elif position == 1 and (close[i] <= ema34_1d_aligned[i] or direction_1w_aligned[i] == -1):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= ema34_1d_aligned[i] or direction_1w_aligned[i] == 1):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_1wSupertrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0