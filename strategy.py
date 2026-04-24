#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend.
- Williams Alligator: Jaw (EMA13 of median price, 8-bar shift), Teeth (EMA8 of median price, 5-bar shift), Lips (EMA5 of median price, 3-bar shift).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and close > 1w EMA50.
         Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and close < 1w EMA50.
- Exit: When Alligator alignment reverses (Lips crosses Teeth or Teeth crosses Jaw).
- Works in bull via buying alignment in uptrend, in bear via selling alignment in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines for given OHLC"""
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).ewm(span=13, adjust=False).mean().shift(8)
    teeth = pd.Series(median_price).ewm(span=8, adjust=False).mean().shift(5)
    lips = pd.Series(median_price).ewm(span=5, adjust=False).mean().shift(3)
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high, low, close)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (need enough for Alligator shifts)
    start_idx = max(13, 20)  # Jaw needs 13+8=21, plus volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(jaw_1d[i]) or 
            np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for alignment signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips_1d[i] > teeth_1d[i] and teeth_1d[i] > jaw_1d[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips_1d[i] < teeth_1d[i] and teeth_1d[i] < jaw_1d[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bullish alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if lips_1d[i] <= teeth_1d[i] or teeth_1d[i] <= jaw_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bearish alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if lips_1d[i] >= teeth_1d[i] or teeth_1d[i] >= jaw_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0