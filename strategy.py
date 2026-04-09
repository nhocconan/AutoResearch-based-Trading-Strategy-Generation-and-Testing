#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v2
# Hypothesis: Breakouts at daily Camarilla pivot levels (H3/L3) with volume confirmation and trend filter using daily EMA(50).
# Only long when price > daily EMA(50), only short when price < daily EMA(50) to avoid counter-trend trades.
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.
# Works in bull markets (trend-following longs) and bear markets (trend-following shorts).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily close, high, low for Camarilla levels
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    camarilla_h3 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_l3 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Daily EMA(50) for trend filter
    close_series = pd.Series(daily_close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
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
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]):
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
            # Enter long: price breaks above H3 with volume confirmation AND above daily EMA(50)
            if close[i] > camarilla_h3_aligned[i] and volume[i] > vol_ma_20[i] * 1.5 and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation AND below daily EMA(50)
            elif close[i] < camarilla_l3_aligned[i] and volume[i] > vol_ma_20[i] * 1.5 and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals