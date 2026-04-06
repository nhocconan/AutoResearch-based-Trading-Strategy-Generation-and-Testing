#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from daily pivot with volume confirmation.
# Goes long when price breaks above R4 with volume > 1.5x average (strong breakout).
# Goes short when price breaks below S4 with volume > 1.5x average.
# Uses S3/R3 as fade zones: short at R3 with volume confirmation, long at S3.
# Uses 1w EMA trend filter to align with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_camarilla1d_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Where C = (H+L+C)/3 (typical price)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and ranges
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + (range_hl * 1.1 / 2)
    r3 = close_1d + (range_hl * 1.1 / 4)
    s3 = close_1d - (range_hl * 1.1 / 4)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Align to 6h timeframe (shifted by 1 day for prior day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1w EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma  # Volume above average
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S3 or 1w EMA turns down significantly
            elif close[i] < s3_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 or 1w EMA turns up significantly
            elif close[i] > r3_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Strong breakout entries with volume confirmation
            if vol_strong[i]:
                # Long breakout: price breaks above R4 with strong volume
                if close[i] > r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below S4 with strong volume
                elif close[i] < s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            # Fade entries at S3/R3 with volume confirmation (counter-trend)
            elif vol_filter[i]:
                # Long at S3: price touches S3 with volume, expecting bounce
                if close[i] <= s3_aligned[i] * 1.001 and close[i] >= s3_aligned[i] * 0.999:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short at R3: price touches R3 with volume, expecting rejection
                elif close[i] >= r3_aligned[i] * 0.999 and close[i] <= r3_aligned[i] * 1.001:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals