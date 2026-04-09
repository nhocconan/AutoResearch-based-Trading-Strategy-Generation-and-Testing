#!/usr/bin/env python3
# 1h_4d_cam_volume_breakout_v1
# Hypothesis: 1-hour breakouts at 4-hour Camarilla pivot levels (H3/L3) with daily volume confirmation (>1.5x 20-bar average volume).
# The 4-hour Camarilla levels provide stronger intraday support/resistance than 1-hour levels.
# Daily volume filter ensures breakouts have institutional participation, reducing false signals.
# Works in bull markets (upward breaks) and bear markets (downward breaks) by following momentum.
# Target: 15-37 trades per year per symbol (~60-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_cam_volume_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h close for Camarilla levels
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_h3_4h = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_l3_4h = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_1h = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Daily volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_1h[i]) or np.isnan(camarilla_l3_1h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L3 level
            if close[i] <= camarilla_l3_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H3 level
            if close[i] >= camarilla_h3_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if close[i] > camarilla_h3_1h[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below L3 with volume confirmation
            elif close[i] < camarilla_l3_1h[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals