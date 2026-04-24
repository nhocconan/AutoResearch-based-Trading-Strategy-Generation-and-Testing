#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Trend filter: 12h EMA34 - price above EMA = bullish trend, below = bearish trend.
- Entry logic:
  - Bullish trend (price > 12h EMA34): Long when Bull Power > 0 AND volume spike (1.5x 20-period MA).
  - Bearish trend (price < 12h EMA34): Short when Bear Power < 0 AND volume spike.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit) OR loss of volume confirmation.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull markets via trend-following breakouts and in bear markets via shorting weakness.
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Elder Ray Index (13-period) on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 34, 20, 13)  # Need enough 12h bars for EMA and 6h for EMA13/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_34_12h_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_close > ema_trend:  # Bullish trend (price > 12h EMA34)
                    # Long when Bull Power > 0 (strong buying pressure)
                    if bull_power[i] > 0:
                        signals[i] = 0.25
                        position = 1
                elif curr_close < ema_trend:  # Bearish trend (price < 12h EMA34)
                    # Short when Bear Power < 0 (strong selling pressure)
                    if bear_power[i] < 0:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR loss of volume confirmation
            if bull_power[i] <= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR loss of volume confirmation
            if bear_power[i] >= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0