#!/usr/bin/env python3
"""
1d_Williams_Alligator_1wTrend_Momentum
Hypothesis: Use Williams Alligator (13/8/5 SMAs) on daily to determine trend,
with 1-week EMA50 as higher timeframe trend filter, and momentum (ROC>0) for entry.
Designed for low trade frequency (<20/year) to minimize fee drag.
Works in bull markets (trend following) and bear markets (mean reversion via Alligator jaws).
"""

name = "1d_Williams_Alligator_1wTrend_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Williams Alligator ===
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period SMMA shifted 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period SMMA shifted 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period SMMA shifted 3
    
    # === 1w EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Momentum Filter (ROC 5-period) ===
    roc = (close - np.roll(close, 5)) / np.roll(close, 5) * 100
    roc[0:5] = np.nan  # First 5 values invalid
    
    # === Volume Filter (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20 * 1.3  # Require 1.3x average volume
    
    # === Position Sizing ===
    position_size = 0.25  # 25% of capital
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Alligator and ROC)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(roc[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator conditions: jaws, teeth, lips alignment
        # Bullish: lips > teeth > jaws (green alignment)
        # Bearish: lips < teeth < jaws (red alignment)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish Alligator + price above 1w EMA50 + positive momentum + volume
            if (bullish_alignment and 
                close[i] > ema50_1w_aligned[i] and 
                roc[i] > 0 and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Bearish Alligator + price below 1w EMA50 + negative momentum + volume
            elif (bearish_alignment and 
                  close[i] < ema50_1w_aligned[i] and 
                  roc[i] < 0 and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Alligator turns bearish OR price crosses below 1w EMA50
                if not bullish_alignment or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: Alligator turns bullish OR price crosses above 1w EMA50
                if not bearish_alignment or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals