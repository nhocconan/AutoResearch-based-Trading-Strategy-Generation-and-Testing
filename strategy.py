#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA trend and daily for Camarilla pivot levels.
- Camarilla levels: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6 from prior 1d bar.
- Breakout long when price > H3, short when price < L3.
- Trend filter: Only trade in direction of 1w EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current volume > 2.0x 50-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying H3 breakouts in uptrend, in bear via selling L3 breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivot levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # H3 = close + 1.1*(high-low)/6
    # L3 = close - 1.1*(high-low)/6
    camarilla_h3_1d = close_1d + (1.1 * (high_1d - low_1d) / 6)
    camarilla_l3_1d = close_1d - (1.1 * (high_1d - low_1d) / 6)
    
    # Align Camarilla levels to 12h (use prior day's levels)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 50-period volume MA
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 34)  # 1d Camarilla + 1w EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1w EMA34 trend
            if i > 0 and not np.isnan(ema_34_1w_aligned[i-1]):
                ema34_slope = ema_34_1w_aligned[i] - ema_34_1w_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Breakout above H3
                    if close[i] > camarilla_h3_1d_aligned[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Breakdown below L3
                    if close[i] < camarilla_l3_1d_aligned[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 or opposite signal
            if close[i] < camarilla_l3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 or opposite signal
            if close[i] > camarilla_h3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0