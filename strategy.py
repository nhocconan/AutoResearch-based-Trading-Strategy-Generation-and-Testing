#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 level and close > 4h EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 level and close < 4h EMA50 (bearish trend)
- Volume must be > 1.8x 24-period average for high-conviction breakouts
- Uses 4h HTF for trend filter to reduce noise and align with higher timeframe structure
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Designed to work in both bull and bear markets via strong trend filter and breakout structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day (24 bars of 1h = 1d)
        if i >= 24:
            prev_high = np.max(high[i-24:i])
            prev_low = np.min(low[i-24:i])
            prev_close = close[i-1]
            
            pivot[i] = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            camarilla_h3[i] = pivot[i] + range_val * 1.1 / 4.0  # H3 level
            camarilla_l3[i] = pivot[i] - range_val * 1.1 / 4.0  # L3 level
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 1.8x 24-period average volume
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50, 24) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(pivot[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume spike
            if close[i] > camarilla_h3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume spike
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L3 (mean reversion) or trend breaks
            if close[i] < camarilla_l3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above Camarilla H3 (mean reversion) or trend breaks
            if close[i] > camarilla_h3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0