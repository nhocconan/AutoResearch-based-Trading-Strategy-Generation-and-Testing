#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 Breakout with 4h EMA50 Trend Filter and Volume Spike.
- Primary timeframe: 1h for execution, HTF: 4h for EMA50 trend direction.
- Camarilla pivot levels (H3, L3) calculated from prior 4h bar: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2.
- Trend filter: 4h EMA50 slope (rising/falling) determines bias.
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Entry: Long when price breaks above H3 in uptrend + volume spike. Short when price breaks below L3 in downtrend + volume spike.
- Exit: Reverse signal or trailing stop via EMA21 cross.
- Session filter: 08-20 UTC to avoid low-activity hours.
- Discrete signal size: 0.20 to minimize fee churn.
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
    
    # Calculate 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla H3 and L3 levels (based on prior 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA50 slope for trend direction
        ema50_slope = ema_50_4h_aligned[i] - ema_50_4h_aligned[i-1]
        
        if position == 0:
            # Look for breakouts in direction of 4h trend
            if ema50_slope > 0 and close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                # Uptrend: buy on break above H3
                signals[i] = 0.20
                position = 1
            elif ema50_slope < 0 and close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                # Downtrend: sell on break below L3
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below L3 or trend reversal
            if close[i] < camarilla_l3_aligned[i] or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 or trend reversal
            if close[i] > camarilla_h3_aligned[i] or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0