#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v3
# Hypothesis: 12h strategy using 1d Camarilla pivot levels for structure, 1d EMA trend filter, and volume confirmation.
# Long when price > EMA50, touches/breaks Camarilla H3 resistance with volume > 1.5x average.
# Short when price < EMA50, touches/breaks Camarilla L3 support with volume > 1.5x average.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First value has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    r2 = pivot + (range_hl * 1.1 / 6)
    r1 = pivot + (range_hl * 1.1 / 12)
    
    # Support levels
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA50 (trend reversal) or below S3 (deep pullback)
            if close[i] < ema_50_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA50 (trend reversal) or above R3 (strong bounce)
            if close[i] > ema_50_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price > EMA50 (uptrend) and breaks above R3 resistance
                if close[i] > ema_50_aligned[i] and close[i] > r3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price < EMA50 (downtrend) and breaks below S3 support
                elif close[i] < ema_50_aligned[i] and close[i] < s3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals