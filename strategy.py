#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and 1d for Camarilla levels.
- Camarilla: H3 = close + 1.1*(high-low)/12, L3 = close - 1.1*(high-low)/12 (from prior day).
- Long when price breaks above H3 with volume spike, short when breaks below L3 with volume spike.
- Trend filter: Only trade in direction of 4h EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
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
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla levels (prior day's high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # H3 = close + 1.1*(high-low)/12, L3 = close - 1.1*(high-low)/12
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 1h (use prior day's levels for current day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 4h EMA50 trend
            if i > 0 and not np.isnan(ema_50_4h_aligned[i-1]):
                ema50_slope = ema_50_4h_aligned[i] - ema_50_4h_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Long when price breaks above H3 with volume spike
                    if close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                        signals[i] = 0.20
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Short when price breaks below L3 with volume spike
                    if close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 or opposite signal
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 or opposite signal
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0