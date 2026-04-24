#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for volume and chop regime.
- Volume confirmation: current 4h volume > 1.5 * 20-period volume MA to filter weak breakouts.
- Choppiness regime: CHOP(14) > 61.8 = ranging (mean revert at H3/L3), CHOP < 38.2 = trending (breakout).
- Entry: In trending (CHOP < 38.2): Long on break above H3, Short on break below L3.
         In ranging (CHOP > 61.8): Long on reversal from L3 (close > low after touch), Short on reversal from H3 (close < high after touch).
- Exit: Opposite H3/L3 breakout or regime shift.
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
    
    # Get 1d data for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period) for choppiness
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d choppiness index (14-period)
    # Sum of true range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Chop = 100 * log10(tr_sum / range_14) / log10(14)
    chop = 100 * np.log10(tr_sum / (range_14 + 1e-10)) / np.log10(14)
    
    # Align 1d volume and chop to 4h
    volume_1d = df_1d['volume'].values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Camarilla levels (H3, L3) from previous 1d OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    # We use the previous completed 1d bar's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6.0
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6.0
    
    # Align Camarilla levels to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for ATR/chop and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
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
            # Long exit: price closes below L3 OR regime shifts to trending (chop < 38.2) from ranging
            if curr_close < l3 or (chop_val < 38.2 and chop_aligned[i-1] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR regime shifts to trending (chop < 38.2) from ranging
            if curr_close > h3 or (chop_val < 38.2 and chop_aligned[i-1] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0