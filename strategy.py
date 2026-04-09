#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Daily breakouts at weekly Camarilla pivot levels (H3/L3) with volume confirmation (>1.5x 50-day average volume) and trend filter using weekly EMA(20).
# Weekly Camarilla levels act as strong support/resistance; breaks signal momentum continuation.
# EMA(20) filter ensures trades align with long-term trend to reduce whipsaws in sideways markets.
# Volume filter reduces false breakouts. Works in bull markets (upward breaks above EMA) and bear markets (downward breaks below EMA).
# Target: 10-25 trades per year per symbol (~40-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly close for Camarilla levels
    weekly_close = df_1w['close'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    camarilla_h3 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
    camarilla_l3 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Trend filter: weekly EMA(20)
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 50-period average (using simple moving average)
    vol_ma_50 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 50:
            vol_sum -= volume[i-50]
        if i >= 49:
            vol_ma_50[i] = vol_sum / 50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_50[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L3 level
            if close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H3 level
            if close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation AND above EMA20 (uptrend)
            if close[i] > camarilla_h3_aligned[i] and volume[i] > vol_ma_50[i] * 1.5 and close[i] > ema_20[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation AND below EMA20 (downtrend)
            elif close[i] < camarilla_l3_aligned[i] and volume[i] > vol_ma_50[i] * 1.5 and close[i] < ema_20[i]:
                position = -1
                signals[i] = -0.25
    
    return signals