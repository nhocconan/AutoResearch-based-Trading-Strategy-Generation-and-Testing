#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v6
# Hypothesis: Weekly trend + Daily Camarilla breakout with volume confirmation.
# Long: Price above weekly EMA(13) and breaks above daily H4, with volume > 1.5x average.
# Short: Price below weekly EMA(13) and breaks below daily L4, with volume > 1.5x average.
# Exit: Return to daily pivot point (mean reversion).
# Uses 1d (primary) and 1w (HTF) timeframes to reduce trades and improve quality.
# Target: 10-25 trades/year (40-100 total over 4 years) with strict entry conditions.
# Works in bull markets (trend continuation) and bear markets (mean reversion to pivot).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v6"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (same as primary for self-reference)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Daily Camarilla levels (using H4/L4 for wider bands = fewer trades)
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    pc_1d = df_1d['close'].values
    
    range_1d = ph_1d - pl_1d
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    h4 = pc_1d + range_1d * 1.1 / 2
    l4 = pc_1d - range_1d * 1.1 / 2
    # Pivot point = (high + low + close) / 3
    pp = (ph_1d + pl_1d + pc_1d) / 3
    
    # Align Daily Camarilla levels to 1d timeframe (wait for previous day's close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate Weekly EMA(13) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w, dtype=float)
    ema_1w[0] = close_1w[0]
    alpha = 2.0 / (13 + 1)
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align Weekly EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > weekly EMA for longs, price < weekly EMA for shorts
        trend_long = close[i] > ema_1w_aligned[i]
        trend_short = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below pivot point (mean reversion)
            if close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above pivot point (mean reversion)
            if close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 with volume confirmation and weekly uptrend
            if close[i] > h4_aligned[i] and vol_ok and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 with volume confirmation and weekly downtrend
            elif close[i] < l4_aligned[i] and vol_ok and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals