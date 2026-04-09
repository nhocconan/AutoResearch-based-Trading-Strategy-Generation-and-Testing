#!/usr/bin/env python3
# 12h_camarilla_1w_trend_volume_v1
# Hypothesis: 12h Camarilla pivot levels from 1w HTF + volume confirmation + 1w EMA50 trend filter.
# Uses 1w timeframe for structure (Camarilla pivots and EMA50) to reduce noise and overtrading.
# Volume confirmation requires current 12h volume > 2.0x 20-period average.
# Discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull/bear by aligning with 1w trend via EMA50. Volume confirms institutional participation.
# ATR-based volatility filter added to avoid low-volatility whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1w_trend_volume_v1"
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
    
    # 1w ATR for volatility filter (ATR14)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
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
    
    # Align to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average (balanced)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility conditions that cause whipsaws
        # Only trade when current ATR > 0.4 * 50-period ATR average (adaptive threshold)
        atr_ma = pd.Series(atr14_1w_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma[i]) or atr14_1w_aligned[i] < 0.4 * atr_ma[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 OR trend turns bearish
            if close[i] < h3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 OR trend turns bullish
            if close[i] > l3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above H3 with bullish trend
                if close[i] > h3_aligned[i] and close[i] > ema50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L3 with bearish trend
                elif close[i] < l3_aligned[i] and close[i] < ema50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals