#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h for execution and signal generation.
- HTF: 1d for volume confirmation and choppiness regime (CHOP > 61.8 = ranging, CHOP < 38.2 = trending).
- In trending markets (CHOP < 38.2): Breakout strategy - Long when price closes above H3, Short when price closes below L3.
- In ranging markets (CHOP > 61.8): Mean reversion - Long when price touches L3 and reverses up, Short when price touches H3 and reverses down.
- Volume confirmation: current 12h volume > 1.5 * 20-period volume MA to avoid false breakouts.
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
    
    # Get 1d data for volume MA and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d True Range for choppiness
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).sum().values  # ATR sum for CHOP
    
    # Calculate 1d high-low range for choppiness
    hl_range = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(ATR_sum / HL_sum) / log10(14)
    chop_raw = 100 * np.log10(atr_1d / hl_range) / np.log10(14)
    chop = pd.Series(chop_raw).fillna(50).values  # fill NaN with 50 (neutral)
    
    # Align 1d volume MA and chop to 12h
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * volume_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels (H3, L3) from previous 1d
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + (1.1 * (prev_high - prev_low) / 6)
    camarilla_l3 = prev_close - (1.1 * (prev_high - prev_low) / 6)
    
    # Align Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike = volume_spike_aligned[i]
        chop_val = chop_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_spike:
                if chop_val < 38.2:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above H3
                    if curr_close > h3:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below L3
                    elif curr_close < l3:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: mean reversion at extremes
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3 and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3 and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR chop shifts to trending (from ranging)
            if curr_close < l3 or (chop_val < 38.2 and chop_aligned[i-1] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR chop shifts to trending (from ranging)
            if curr_close > h3 or (chop_val < 38.2 and chop_aligned[i-1] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dVolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0