#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter with volume confirmation.
- Primary timeframe: 12h for lower trade frequency and better generalization.
- Williams Alligator: Jaw (EMA13, 8-offset), Teeth (EMA8, 5-offset), Lips (EMA5, 3-offset).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5 * 20-period MA.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5 * 20-period MA.
- Exit: When Alligator alignment reverses or price crosses 1d EMA50.
- Works in bull via buying bullish Alligator alignment in uptrend, in bear via selling bearish alignment in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator on 12h
    # Jaw: EMA13, 8-period offset
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # 8-bar offset into future
    jaw[:8] = np.nan  # First 8 values invalid
    
    # Teeth: EMA8, 5-period offset
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # 5-bar offset into future
    teeth[:5] = np.nan  # First 5 values invalid
    
    # Lips: EMA5, 3-period offset
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # 3-bar offset into future
    lips[:3] = np.nan  # First 3 values invalid
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 20, 13)  # Need enough 1d bars for EMA50 and Alligator offsets
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Alligator alignment signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    # Long: price above 1d EMA50 (uptrend)
                    if close[i] > ema_50_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    # Short: price below 1d EMA50 (downtrend)
                    if close[i] < ema_50_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator alignment reverses or price crosses below EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment reverses or price crosses above EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0