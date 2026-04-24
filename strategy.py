#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA50 trend direction.
- Bull Power = High - EMA13, Bear Power = Low - EMA13 (EMA13 on 6h).
- In uptrend (price > 12h EMA50): Long when Bull Power > 0 and rising (2-bar momentum) AND volume > 1.5 * 20-period volume MA.
- In downtrend (price < 12h EMA50): Short when Bear Power < 0 and falling (2-bar momentum) AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite signal or trend reversal (price crosses 12h EMA50).
- Volume confirmation avoids low-conviction moves.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull markets via trend-following longs, in bear markets via trend-following shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 on 6h for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    df_12h_close = pd.Series(df_12h['close'])
    ema50_12h = df_12h_close.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 50, 20)  # Need enough bars for EMA13, EMA50 alignment, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        ema50_val = ema50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                # Uptrend: price above 12h EMA50 -> look for long
                if curr_close > ema50_val:
                    # Bull Power > 0 and rising (current > previous)
                    if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                        signals[i] = 0.25
                        position = 1
                # Downtrend: price below 12h EMA50 -> look for short
                elif curr_close < ema50_val:
                    # Bear Power < 0 and falling (current < previous)
                    if bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price crosses below 12h EMA50 OR Bull Power becomes negative
            if curr_close < ema50_val or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 12h EMA50 OR Bear Power becomes positive
            if curr_close > ema50_val or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0