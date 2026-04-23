#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA Trend Filter and Volume Confirmation
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 12h EMA13)
- Long when Bull Power > 0 and Bear Power < 0 (both bullish) + price above 12h EMA50 + volume > 1.5x 20-period average
- Short when Bear Power > 0 and Bull Power < 0 (both bearish) + price below 12h EMA50 + volume > 1.5x 20-period average
- Uses 6h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in bull markets via trend continuation, in bear markets via counter-trend reversals at extremes
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
    
    # Get 12h data for EMA calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA13 for Elder Ray
    ema13_12h = pd.Series(df_12h['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe (completed 12h bar only)
    ema13_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Elder Ray components for current 6h bar
        bull_power = high[i] - ema13_aligned[i]
        bear_power = ema13_aligned[i] - low[i]
        
        # Determine trend direction from 12h EMA50
        # Use 12h close price for trend determination
        close_12h = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema50_aligned[i]
        downtrend = close_12h_aligned[i] < ema50_aligned[i]
        
        # Elder Ray signals with trend filter and volume confirmation
        # Long: Bull Power > 0 AND Bear Power < 0 (both bullish) + uptrend + volume spike
        # Short: Bear Power > 0 AND Bull Power < 0 (both bearish) + downtrend + volume spike
        long_signal = (bull_power > 0 and 
                      bear_power < 0 and
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (bear_power > 0 and 
                       bull_power < 0 and
                       downtrend and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray divergence or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power becomes positive (bearish pressure) or trend turns down
                if (bear_power > 0 or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bull Power becomes positive (bullish pressure) or trend turns up
                if (bull_power > 0 or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0