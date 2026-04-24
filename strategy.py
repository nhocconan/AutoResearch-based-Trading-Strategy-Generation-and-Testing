#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA50 trend direction.
- EMA50 > rising indicates bullish bias, EMA50 < falling indicates bearish bias.
- Entry: Long when price breaks above Camarilla H3 AND EMA50 trending up.
         Short when price breaks below Camarilla L3 AND EMA50 trending down.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short).
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
    
    # Get 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Get 1d data for Camarilla pivot levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA50 and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema50_val = ema50_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        # Determine EMA50 trend direction (using previous bar)
        if i > start_idx:
            ema50_prev = ema50_aligned[i-1]
            ema50_rising = ema50_val > ema50_prev
            ema50_falling = ema50_val < ema50_prev
        else:
            ema50_rising = False
            ema50_falling = False
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 with rising EMA50
                if curr_high > h3 and ema50_rising:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 with falling EMA50
                elif curr_low < l3 and ema50_falling:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 (opposite level)
            if curr_low < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 (opposite level)
            if curr_high > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0