#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d with volume confirmation.
# Strategy fades at R3/S3 levels (mean reversion) and breaks out at R4/S4 (trend continuation).
# Uses 1d OHLC to calculate Camarilla levels, aligned to 6bars.
# Volume filter ensures sufficient participation for reliable signals.
# Works in both bull (breakout continuation) and bear (mean reversion at extremes) markets.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled risk.

name = "6h_camarilla1d_vol_v2"
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
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * range_ / 6
    camarilla_s3 = prev_close - 1.1 * range_ / 6
    camarilla_r4 = prev_close + 1.1 * range_ / 2
    camarilla_s4 = prev_close - 1.1 * range_ / 2
    
    # Align Camarilla levels to 6bars (shifted by 1 day for prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) or breaks below S4 (trend reversal)
            if close[i] <= s3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) or breaks above R4 (trend reversal)
            if close[i] >= r3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Fade at R3/S3: sell at R3, buy at S3 (mean reversion)
                if close[i] >= r3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                elif close[i] <= s3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Breakout continuation at R4/S4: buy above R4, sell below S4
                elif close[i] > r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals