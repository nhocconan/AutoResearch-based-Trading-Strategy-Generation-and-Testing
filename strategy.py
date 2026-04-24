#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- Entry: Long when price breaks above Camarilla H3 AND 12h EMA34 is rising (close > prior EMA).
         Short when price breaks below Camarilla L3 AND 12h EMA34 is falling (close < prior EMA).
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false breakouts.
- Camarilla levels calculated from prior 12h bar's high-low range.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
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
    
    # Get 12h data for Camarilla calculation and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) from 12h
    # Based on prior 12h bar's high-low range
    h12 = df_12h['high'].values
    l12 = df_12h['low'].values
    c12 = df_12h['close'].values
    
    # Camarilla width
    rang = h12 - l12
    h3 = c12 + (rang * 1.1 / 4)
    l3 = c12 - (rang * 1.1 / 4)
    
    # Calculate 12h EMA34 for trend filter
    ema_34 = pd.Series(c12).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_rising = ema_34 > np.roll(ema_34, 1)  # True if current > prior
    # Handle first element
    ema_rising[0] = False
    
    # Align 12h indicators to 6h
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need enough 12h bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_up = ema_rising_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 with rising EMA
                if curr_high > h3_val and ema_up:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 with falling EMA
                elif curr_low < l3_val and not ema_up:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA trend turns down
            if curr_low < l3_val or not ema_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA trend turns up
            if curr_high > h3_val or ema_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0