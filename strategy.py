#!/usr/bin/env python3
# 4h_camarilla_pivot_12h_volume_v1
# Hypothesis: 4h strategy using 1d Camarilla pivot levels for entry/exit,
# with 12h volume confirmation to filter false breakouts. Only trade when
# price touches Camarilla H3/L3 levels with volume > 2.0x 20-period 12h average.
# Exit at H4/L4 levels or opposite H3/L3 touch. Uses discrete sizing (0.0, ±0.25)
# to minimize fee churn. Target: 20-40 trades/year. Works in bull/bear via
# mean-reversion at extreme pivot levels (H3/L3 act as support/resistance).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla pivot levels for previous day
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels (H3/L3 for entry, H4/L4 for exit)
    h3 = pivot + (range_val * 1.1 / 4.0)  # ~1.1 * range / 4
    l3 = pivot - (range_val * 1.1 / 4.0)
    h4 = pivot + (range_val * 1.1 / 2.0)  # ~1.1 * range / 2
    l4 = pivot - (range_val * 1.1 / 2.0)
    
    # Align daily Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 12h HTF data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (approximate using aligned MA as proxy)
        # For volume confirmation, we check if current 4h volume is above 2.0x 12h volume MA
        volume_s = pd.Series(volume)
        volume_ma_4h = volume_s.rolling(window=20, min_periods=20).mean().values
        if np.isnan(volume_ma_4h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 2.0 * volume_ma_4h[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H4 (take profit) or touches L3 (stop/reverse)
            if close[i] >= h4_aligned[i] or close[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L4 (take profit) or touches H3 (stop/reverse)
            if close[i] <= l4_aligned[i] or close[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price touches L3 with volume confirmation
                if close[i] <= l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches H3 with volume confirmation
                elif close[i] >= h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals