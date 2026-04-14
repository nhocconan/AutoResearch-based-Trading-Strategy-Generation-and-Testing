#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 1d Trend Filter and Volume Confirmation
# Uses daily Camarilla pivot levels for breakout entries
# 1d EMA (50) provides trend filter to avoid counter-trend trades
# Volume > 1.5x average confirms breakout strength
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drift
# Works in bull/bear markets by combining structure (pivots) with trend and volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3 (primary breakout levels)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_h3 = pivot + (range_1d * 1.1 / 2)
    camarilla_l3 = pivot - (range_1d * 1.1 / 2)
    
    # Calculate 1d EMA (50) for trend direction
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d data to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above Camarilla H3 with volume and uptrend
            if price > camarilla_h3_aligned[i] and vol > 1.5 * vol_ma[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below Camarilla L3 with volume and downtrend
            elif price < camarilla_l3_aligned[i] and vol > 1.5 * vol_ma[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Camarilla pivot or trend changes
            if price < camarilla_l3_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Camarilla pivot or trend changes
            if price > camarilla_h3_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0