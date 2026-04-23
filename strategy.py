#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume spike (>2.0x average).
- Uses 12h for trend filter (EMA50) to reduce whipsaw in bear markets
- Volume spike reduces false breakouts (strict >2.0x to limit trades)
- Uses 1d for Camarilla pivot levels (structure from higher timeframe)
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 24-period average (strict for 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d Camarilla pivot levels (H3, L3, H4, L4)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    range_1d = high_1d - low_1d
    # Camarilla levels
    h3_1d = pp_1d + range_1d * 1.1 / 4
    l3_1d = pp_1d - range_1d * 1.1 / 4
    h4_1d = pp_1d + range_1d * 1.1 / 2
    l4_1d = pp_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use prior completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up_h3 = close[i] > h3_1d_aligned[i-1]  # Close above prior 1d H3
        breakout_down_l3 = close[i] < l3_1d_aligned[i-1]  # Close below prior 1d L3
        
        if position == 0:
            # Long: 1d Camarilla H3 breakout up AND price > 12h EMA50 AND volume confirmation
            if breakout_up_h3 and volume_confirm and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1d Camarilla L3 breakout down AND price < 12h EMA50 AND volume confirmation
            elif breakout_down_l3 and volume_confirm and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Camarilla L4 break down OR price < 12h EMA50 (trend flip)
            if close[i] < l4_1d_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Camarilla H4 break up OR price > 12h EMA50 (trend flip)
            if close[i] > h4_1d_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeSpike_Strict"
timeframe = "4h"
leverage = 1.0