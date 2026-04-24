#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA34 trend direction.
- EMA34 > rising indicates bullish trend, EMA34 < falling indicates bearish trend.
- Entry: Long when price breaks above H3 level AND EMA34 trending up AND volume spike.
         Short when price breaks below L3 level AND EMA34 trending down AND volume spike.
         In weak trend (EMA34 flat): fade at H4/L4 levels with reversal confirmation.
- Exit: Opposite H3/L3 breakout or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Get 1d data for EMA34 and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate EMA34 slope for trend direction (3-period change)
    ema_34_slope = np.zeros_like(ema_34)
    ema_34_slope[3:] = (ema_34[3:] - ema_34[:-3]) / 3  # 3-period slope
    
    # Align 1d EMA34 and slope to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    h3 = prev_close + prev_range * 1.1 / 4
    l3 = prev_close - prev_range * 1.1 / 4
    h4 = prev_close + prev_range * 1.1 / 2
    l4 = prev_close - prev_range * 1.1 / 2
    
    # Align Camarilla levels to 6h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        ema_34_slope_val = ema_34_slope_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema_34_slope_val > 0.0001:  # Strong bullish trend
                    # Bullish breakout: price breaks above H3
                    if curr_high > h3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema_34_slope_val < -0.0001:  # Strong bearish trend
                    # Bearish breakout: price breaks below L3
                    if curr_low < l3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Weak trend (EMA34 flat): mean reversion at H4/L4
                    # Fade at H4: price reaches H4 and shows reversal (close < high)
                    if curr_high >= h4_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
                    # Fade at L4: price reaches L4 and shows reversal (close > low)
                    elif curr_low <= l4_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA trend turns bearish
            if curr_low < l3_aligned[i] or ema_34_slope_val < -0.0001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA trend turns bullish
            if curr_high > h3_aligned[i] or ema_34_slope_val > 0.0001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_CamarillaH3L3_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0