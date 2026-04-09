#!/usr/bin/env python3
# 1d_camarilla_1w_trend_volume_v1
# Hypothesis: Daily strategy using weekly Camarilla H3/L3 levels with 1-week EMA trend filter and volume confirmation.
# H3/L3 levels provide strong reversal zones. Weekly EMA(20) determines trend bias. Volume > 2x 20-period average
# filters weak breakouts. Discrete sizing (±0.25) minimizes fee churn. Designed to work in both bull and bear markets
# by aligning with weekly trend. Target: 30-100 total trades over 4 years (7-25/year).

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
    
    # 1-week HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1-week EMA(20) for trend bias
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily HTF data for Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    
    # Camarilla H3 and L3 levels (strong reversal levels)
    h3 = pivot_point + (range_1d * 1.1 / 4)
    l3 = pivot_point - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back below H3 (profit taken or reversal)
            if close[i] < h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back above L3 (profit taken or reversal)
            if close[i] > l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price crosses above H3 with weekly bullish bias (price above EMA20)
                if close[i] > h3_aligned[i] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below L3 with weekly bearish bias (price below EMA20)
                elif close[i] < l3_aligned[i] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals