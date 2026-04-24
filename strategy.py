#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d EMA50 trend + volume spike confirmation.
- Williams Alligator uses smoothed medians (Jaw, Teeth, Lips) to identify trend and entry points.
- Long when Lips > Teeth > Jaw (bullish alignment) and price > 1d EMA50 with volume confirmation.
- Short when Lips < Teeth < Jaw (bearish alignment) and price < 1d EMA50 with volume confirmation.
- Uses 6h timeframe to capture medium-term swings with reduced noise.
- Volume confirmation (>1.5x 20-bar average) filters false signals in low-volatility periods.
- Discrete position size 0.25 balances drawdown control and profit potential.
- Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
- Designed to work in both bull (trend following) and bear (counter-trend alignment via Alligator) markets.
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
    
    # Get 1d data ONCE before loop for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h data (smoothed medians)
    # Jaw: Smoothed Median (13, 8) -> Blue line
    # Teeth: Smoothed Median (8, 5) -> Red line  
    # Lips: Smoothed Median (5, 3) -> Green line
    median = (high + low) / 2
    
    # Jaw: 13-period SMMA, then 8-period SMMA
    jaw_smma1 = pd.Series(median).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw_smma1).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    
    # Teeth: 8-period SMMA, then 5-period SMMA
    teeth_smma1 = pd.Series(median).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth_smma1).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Lips: 5-period SMMA, then 3-period SMMA
    lips_smma1 = pd.Series(median).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips_smma1).ewm(alpha=1/3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    # Long: price above 1d EMA50
                    if close[i] > ema_50_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    # Short: price below 1d EMA50
                    if close[i] < ema_50_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: bearish alignment OR price crosses below 1d EMA50
            if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR price crosses above 1d EMA50
            if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0