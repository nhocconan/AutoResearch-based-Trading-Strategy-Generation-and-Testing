#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for EMA50 trend alignment.
- Williams Alligator: Jaw (EMA13 smoothed 8), Teeth (EMA8 smoothed 5), Lips (EMA5 smoothed 3) on 12h.
- Trend filter: only long when 12h close > 1d EMA50, only short when 12h close < 1d EMA50.
- Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Exit: Alligator lines cross (Lips below Teeth for long exit, Lips above Teeth for short exit).
- Works in bull via trend alignment, in bear via Alligator reversal signals.
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
    
    # Calculate Williams Alligator on 12h
    # Jaw: EMA13 of median price, smoothed by 8 periods
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw_raw.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA8 of median price, smoothed by 5 periods
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth_raw.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA5 of median price, smoothed by 3 periods
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips_raw.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13, 8, 5)  # Need 1d EMA50, volume MA, Alligator components
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND uptrend AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND downtrend AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips crosses below Teeth (bullish momentum fading)
            if lips[i] <= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips crosses above Teeth (bearish momentum fading)
            if lips[i] >= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0