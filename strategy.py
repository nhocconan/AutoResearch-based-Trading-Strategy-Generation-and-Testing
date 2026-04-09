#!/usr/bin/env python3
# 1d_camarilla_1w_trend_volume_v1
# Hypothesis: Daily strategy using weekly Camarilla H4/L4 levels with 1w trend filter and volume confirmation.
# In ranging markets, price tends to revert from H4/L4 levels; in trending markets, breaks above/below
# these levels with volume continuation signal strong moves. 1w EMA(50) determines trend bias.
# Volume > 1.5x 20-period average filters weak breakouts. Discrete sizing (±0.30) minimizes fee churn.
# Target: 30-100 total trades over 4 years (7-25/year). Works in both bull and bear via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA(50) for trend bias
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d HTF data for Camarilla pivots (using 1d to calculate pivots, aligned to 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # Handle first bar
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Camarilla pivot point
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla H4 and L4 levels (strongest support/resistance)
    h4 = pivot_point + (range_1d * 1.1 / 2)
    l4 = pivot_point - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back below H4 (profit taken or reversal)
            if close[i] < h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes back above L4 (profit taken or reversal)
            if close[i] > l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price crosses above H4 with 1w bullish bias (price above EMA50)
                if close[i] > h4_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price crosses below L4 with 1w bearish bias (price below EMA50)
                elif close[i] < l4_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals