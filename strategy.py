#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength.
- Long when Lips > Teeth > Jaw (bullish alignment) with volume > 2.0x 20-bar average.
- Short when Lips < Teeth < Jaw (bearish alignment) with volume > 2.0x 20-bar average.
- Trend filter: price must be above/below 1d EMA50 to align with higher timeframe direction.
- Designed for 12h timeframe to capture medium-term trends with low frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Combines proven Williams Alligator with 1d trend filter and volume confirmation.
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
    
    # Get 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Williams Alligator on 12h timeframe
    close_12h = df_12h['close'].values
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe (wait for 12h bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need enough for EMA, volume MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms signal
            if volume_confirm:
                # Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50
                if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50
                elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Lips <= Teeth OR price < 1d EMA50
            if lips_aligned[i] <= teeth_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips >= Teeth OR price > 1d EMA50
            if lips_aligned[i] >= teeth_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0