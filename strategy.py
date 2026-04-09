#!/usr/bin/env python3
# 1d_camarilla_1w_trend_volume_v1
# Hypothesis: 1d Camarilla pivot levels from 1w HTF + volume confirmation + 1w EMA50 trend filter.
# Camarilla pivots provide institutional support/resistance levels; volume confirms participation;
# 1w EMA50 defines long-term trend to avoid counter-trend entries. Works in bull/bear by aligning with HTF trend.
# Target: 7-25 trades/year (30-100 total over 4 years).

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
    
    # 1w HTF data for Camarilla pivots and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w Camarilla pivot levels (based on previous week's range)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Calculated from previous week's high, low, close
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan  # First value has no previous
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels
    h4 = pivot_point + (range_1w * 1.1 / 2)
    h3 = pivot_point + (range_1w * 1.1 / 4)
    h2 = pivot_point + (range_1w * 1.1 / 6)
    h1 = pivot_point + (range_1w * 1.1 / 12)
    l1 = pivot_point - (range_1w * 1.1 / 12)
    l2 = pivot_point - (range_1w * 1.1 / 6)
    l3 = pivot_point - (range_1w * 1.1 / 4)
    l4 = pivot_point - (range_1w * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    h2_aligned = align_htf_to_ltf(prices, df_1w, h2)
    h1_aligned = align_htf_to_ltf(prices, df_1w, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1w, l1)
    l2_aligned = align_htf_to_ltf(prices, df_1w, l2)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 (strong resistance) OR trend turns bearish
            if close[i] < h3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 (strong support) OR trend turns bullish
            if close[i] > l3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.8 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above H3 with bullish trend (intraday continuation)
                if close[i] > h3_aligned[i] and close[i] > ema50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L3 with bearish trend (intraday continuation)
                elif close[i] < l3_aligned[i] and close[i] < ema50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals