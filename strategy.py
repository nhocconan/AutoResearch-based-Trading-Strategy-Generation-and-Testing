# 1/1/2025
# Hypothesis: 6h Camarilla Pivot Breakout with 12h Trend and Volume Spike
# Camarilla levels derived from 12h bars (not daily) provide tighter intraday structure
# Breakout above H3 or below L3 with volume confirms institutional participation
# 12h EMA50 trend filter ensures trades align with higher timeframe momentum
# Works in bull markets (buy H3 breaks in uptrend) and bear markets (sell L3 breaks in downtrend)
# Target: 20-50 trades/year to avoid fee drag
# Position size: 0.25 (25% of capital)

#!/usr/bin/env python3
name = "6h_Camarilla_H3L3_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Camarilla pivot levels from 12h data (using typical price)
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_12h + low_12h + close_12h) / 3
    range_hl = high_12h - low_12h
    
    # Camarilla levels (H3/L3 are key breakout levels)
    h3 = close_12h + (range_hl * 1.1 / 4)
    l3 = close_12h - (range_hl * 1.1 / 4)
    h4 = close_12h + (range_hl * 1.1 / 2)
    l4 = close_12h - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > h3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume and 12h downtrend
            elif close[i] < l3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below H3 or volume drops
            if close[i] < h3_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above L3 or volume drops
            if close[i] > l3_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla H3/L3 breakout with 12h trend and volume confirmation
# - H3 (resistance) and L3 (support) are key Camarilla breakout levels from 12h data
# - Breakout above H3 with volume in 12h uptrend = long opportunity
# - Breakdown below L3 with volume in 12h downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Trend filter ensures alignment with higher timeframe momentum
# - Works in bull markets (buy H3 breaks in uptrend) and bear markets (sell L3 breaks in downtrend)
# - Exit when price returns to H3/L3 or volume weakens
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Camarilla levels from 12h provide tighter intraday structure than daily levels