#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction.
- EMA34 > EMA50 on 12h indicates bullish trend, EMA34 < EMA50 indicates bearish trend.
- Entry: Long when price breaks above Camarilla H3 level AND 12h EMA34 > EMA50 (bullish breakout in uptrend).
         Short when price breaks below Camarilla L3 level AND 12h EMA34 < EMA50 (bearish breakout in downtrend).
         In neutral trend (EMA34 ≈ EMA50): no new entries, only manage existing positions.
- Exit: Opposite Camarilla level touch (L3 for long, H3 for short) or EMA trend reversal.
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
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate EMAs on 12h
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 4h
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous day's OHLC (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/4
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h (they update daily)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need enough 12h bars for EMAs and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_12h_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 AND EMA34 > EMA50 (uptrend)
                if curr_high > h3_level and ema34_val > ema50_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND EMA34 < EMA50 (downtrend)
                elif curr_low < l3_level and ema34_val < ema50_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches L3 level OR EMA trend turns bearish
            if curr_low <= l3_level or ema34_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches H3 level OR EMA trend turns bullish
            if curr_high >= h3_level or ema34_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0