#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution.
- Williams %R(14) identifies overbought/oversold conditions on 6h.
- 12h EMA(50) determines trend: price > EMA = bullish bias (long signals), price < EMA = bearish bias (short signals).
- Volume confirmation: current 6h volume > 1.8 * 20-period volume MA to avoid false reversals.
- Entry logic:
  * In bullish trend (price > 12h EMA): Long when Williams %R crosses above -80 from below (oversold bounce).
  * In bearish trend (price < 12h EMA): Short when Williams %R crosses below -20 from above (overbought rejection).
- Exit: Opposite Williams %R cross (%R crosses above -20 for long exit, below -80 for short exit) or trend reversal.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets: trend filter adapts bias, Williams %R captures mean reversion within trend.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Williams %R(14) on 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough for EMA, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_50_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_williams = williams_r[i-1]
        curr_williams = williams_r[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_close > ema_val:  # Bullish trend: look for oversold bounces
                    # Long when Williams %R crosses above -80 from below
                    if prev_williams <= -80 and curr_williams > -80:
                        signals[i] = 0.25
                        position = 1
                elif curr_close < ema_val:  # Bearish trend: look for overbought rejections
                    # Short when Williams %R crosses below -20 from above
                    if prev_williams >= -20 and curr_williams < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR trend turns bearish
            if curr_williams >= -20 or curr_close < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR trend turns bullish
            if curr_williams <= -80 or curr_close > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0