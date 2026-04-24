#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Camarilla levels.
- EMA34 > rising indicates bullish trend, EMA34 < falling indicates bearish trend.
- Entry: Long when price breaks above Camarilla H3 AND EMA34 trending up.
         Short when price breaks below Camarilla L3 AND EMA34 trending down.
         In ranging (EMA34 flat): Long at L3 reversal, Short at H3 reversal.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA34 slope for trend direction (using 3-bar change)
    ema34_slope = np.zeros_like(ema34_aligned)
    ema34_slope[3:] = (ema34_aligned[3:] - ema34_aligned[:-3]) / 3
    
    # Calculate Camarilla levels (H3, L3) from previous 1d bar
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # Using previous completed 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 34)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema34_slope[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Trend filter: EMA34 slope
                if ema34_slope[i] > 0.0001:  # Bullish trend
                    # Long breakout above H3
                    if curr_close > h3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope[i] < -0.0001:  # Bearish trend
                    # Short breakdown below L3
                    if curr_close < l3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging (EMA34 flat): mean reversion at extremes
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR EMA trend turns bearish
            if curr_close < l3_aligned[i] or ema34_slope[i] < -0.0001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR EMA trend turns bullish
            if curr_close > h3_aligned[i] or ema34_slope[i] > 0.0001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0