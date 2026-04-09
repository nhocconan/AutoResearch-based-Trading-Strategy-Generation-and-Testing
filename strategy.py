#!/usr/bin/env python3
# 1d_camarilla_1w_trend_volume_v1
# Hypothesis: 1d Camarilla pivot levels from 1w HTF + volume confirmation + 1w EMA50 trend filter.
# Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Target: 20-50 trades/year.
# Works in bull/bear by aligning with 1w trend via EMA50. Volume confirms institutional participation.
# Timeframe: 1d (daily bars) - lower trade frequency reduces fee drag.

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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w Camarilla pivot levels (based on previous week's range)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    pivot_point = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels: H3, L3 (strongest intraday support/resistance)
    h3 = pivot_point + (range_1w * 1.1 / 4)
    l3 = pivot_point - (range_1w * 1.1 / 4)
    
    # Align to 1d timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 OR trend turns bearish
            if close[i] < h3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 OR trend turns bullish
            if close[i] > l3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above H3 with bullish trend
                if close[i] > h3_aligned[i] and close[i] > ema50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price breaks below L3 with bearish trend
                elif close[i] < l3_aligned[i] and close[i] < ema50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals