#!/usr/bin/env python3
# 6h_camarilla_1d_trend_volume_v2
# Hypothesis: 6h strategy using 1d Camarilla pivot levels (H3/L3 for mean reversion, H4/L4 for breakout) with 1w trend filter and volume confirmation.
# In ranging markets (price between H3/L3), fade extremes at H3/L3 with target at pivot point.
# In trending markets (price outside H4/L4), continue breakout in direction of 1w trend.
# Volume > 1.3x 20-period average confirms participation. Discrete sizing (±0.25) to minimize fees.
# Target: 60-120 total trades over 4 years (15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_1d_trend_volume_v2"
timeframe = "6h"
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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d HTF data for Camarilla pivots
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
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Camarilla pivot point
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    h3 = pivot_point + (range_1d * 1.1 / 4)  # Strong resistance
    l3 = pivot_point - (range_1d * 1.1 / 4)  # Strong support
    h4 = pivot_point + (range_1d * 1.1 / 2)  # Breakout resistance
    l4 = pivot_point - (range_1d * 1.1 / 2)  # Breakout support
    
    # Align all levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pivot_point_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if close[i] < pivot_point_aligned[i]:  # Failed to hold above pivot
                position = 0
                signals[i] = 0.0
            elif close[i] > h4_aligned[i] and close_1w[-1] < ema_50_1w[-1]:  # Weekly trend turned bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if close[i] > pivot_point_aligned[i]:  # Failed to hold below pivot
                position = 0
                signals[i] = 0.0
            elif close[i] < l4_aligned[i] and close_1w[-1] > ema_50_1w[-1]:  # Weekly trend turned bullish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.3 * volume_ma[i]
            
            if volume_confirmed:
                # Determine weekly trend bias
                weekly_bullish = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False
                weekly_bearish = close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False
                
                # Mean reversion mode: price between H3 and L3
                if l3_aligned[i] < close[i] < h3_aligned[i]:
                    # Long near L3 support
                    if close[i] <= l3_aligned[i] * 1.002:  # Within 0.2% of L3
                        position = 1
                        signals[i] = 0.25
                    # Short near H3 resistance
                    elif close[i] >= h3_aligned[i] * 0.998:  # Within 0.2% of H3
                        position = -1
                        signals[i] = -0.25
                # Breakout mode: price outside H4/L4
                else:
                    # Long breakout above H4 with weekly bullish bias
                    if close[i] > h4_aligned[i] and weekly_bullish:
                        position = 1
                        signals[i] = 0.25
                    # Short breakout below L4 with weekly bearish bias
                    elif close[i] < l4_aligned[i] and weekly_bearish:
                        position = -1
                        signals[i] = -0.25
    
    return signals