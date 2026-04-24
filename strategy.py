#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend direction.
- EMA34 > price = bullish bias, EMA34 < price = bearish bias.
- Entry: Long when price breaks above Camarilla H3 AND EMA34 > price (bullish breakout in uptrend).
         Short when price breaks below Camarilla L3 AND EMA34 < price (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or EMA trend flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use H3/L3 for breakout entries
    range_1d = high_1d - low_1d
    H3 = close_1d + 1.125 * range_1d
    L3 = close_1d - 1.125 * range_1d
    H4 = close_1d + 1.5 * range_1d
    L4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3.values)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4.values)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4.values)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema34_val > curr_close:  # Bullish bias: EMA above price
                    # Bullish breakout: price breaks above H3
                    if curr_high > H3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_val < curr_close:  # Bearish bias: EMA below price
                    # Bearish breakout: price breaks below L3
                    if curr_low < L3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA flips to bearish
            if curr_low < L3_aligned[i] or ema34_val < curr_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA flips to bullish
            if curr_high > H3_aligned[i] or ema34_val > curr_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0