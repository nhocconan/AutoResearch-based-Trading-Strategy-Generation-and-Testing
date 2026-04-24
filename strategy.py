#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1w EMA50 trend filter + volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA50 trend filter.
- Williams Alligator: Jaw (EMA13,8), Teeth (EMA8,5), Lips (EMA5,3).
  Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish).
- Trend filter: Only trade long when price > 1w EMA50, short when price < 1w EMA50.
- Volume confirmation: current volume > 1.3x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying aligned Alligator in uptrend, in bear via selling aligned Alligator in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 6h
    # Jaw: EMA13, 8 periods offset
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]])  # offset by 8
    
    # Teeth: EMA8, 5 periods offset
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]])  # offset by 5
    
    # Lips: EMA5, 3 periods offset
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]])  # offset by 3
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 21, 20)  # EMA50(1w) + Alligator jaw offset + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Trend filter from 1w EMA50
            price_above_trend = close[i] > ema_50_1w_aligned[i]
            price_below_trend = close[i] < ema_50_1w_aligned[i]
            
            if bullish_alignment and price_above_trend and volume_spike[i]:
                # Bullish Alligator in uptrend: go long
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and price_below_trend and volume_spike[i]:
                # Bearish Alligator in downtrend: go short
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment or price crosses below Jaw
            if lips[i] <= teeth[i] or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment or price crosses above Jaw
            if lips[i] >= teeth[i] or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0