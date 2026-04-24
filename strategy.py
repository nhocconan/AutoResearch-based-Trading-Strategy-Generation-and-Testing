#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and 1d chop regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for Camarilla levels, volume, and chop.
- Camarilla H3 (resistance) and L3 (support) act as institutional pivot levels.
- In choppy market (CHOP > 61.8): mean reversion at H3/L3 (short at H3, long at L3).
- In trending market (CHOP < 38.2): breakout strategy (long above H3, short below L3).
- Volume confirmation: current 12h volume > 2.0 * 20-period 1d volume MA (scaled to 12h).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull/bear: mean reversion in ranging markets, breakout in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, volume MA, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) from previous 1d bar
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    range_ = prev_high - prev_low
    
    # Camarilla levels: H3 = prev_close + 1.1 * range_/6, L3 = prev_close - 1.1 * range_/6
    h3 = prev_close + 1.1 * range_ / 6.0
    l3 = prev_close - 1.1 * range_ / 6.0
    
    # Align Camarilla levels to 12h (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume MA (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Choppiness Index (CHOP) on 1d
    # CHOP = 100 * log10(sum(ATR1) / (n * (HHV - LLV))) / log10(n)
    # We'll use a simplified version: CHOP = 100 * log10(ATR_sum / (n * (HHV - LLV))) / log10(n)
    # But for efficiency, we'll use: CHOP > 61.8 = choppy, CHOP < 38.2 = trending
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    hh_ll_diff = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_sum / (14 * hh_ll_diff + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1d volume MA (scaled)
    # Scale 1d volume MA to 12h: 1d volume represents 2x 12h bars (since 24h/12h=2)
    volume_ma_scaled = volume_ma_1d_aligned * 2.0  # Approximate 12h volume expectation
    volume_spike = volume > volume_ma_scaled
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough 1d bars for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                if chop_val > 61.8:  # Choppy/ranging regime: mean reversion at H3/L3
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3_val and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3_val and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
                elif chop_val < 38.2:  # Trending regime: breakout strategy
                    # Long when price breaks above H3
                    if curr_close > h3_val:
                        signals[i] = 0.25
                        position = 1
                    # Short when price breaks below L3
                    elif curr_close < l3_val:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price crosses below L3 (mean reversion) or H3 (trending breakout failed)
            if chop_val > 61.8:  # In chop: exit at opposite level (L3)
                if curr_close < l3_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # In trend: exit if price fails to hold above H3
                if curr_close < h3_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above H3 (mean reversion) or L3 (trending breakout failed)
            if chop_val > 61.8:  # In chop: exit at opposite level (H3)
                if curr_close > h3_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # In trend: exit if price fails to hold below L3
                if curr_close > l3_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_CamarillaH3L3_1dChopRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0