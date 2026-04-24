#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for volume and chop regime.
- Camarilla H3 (resistance) and L3 (support) from prior 1d candle act as intraday pivot levels.
- Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA to filter weak breakouts.
- Chop regime: 1d chop > 61.8 (ranging) enables mean reversion at H3/L3; chop < 38.2 (trending) enables breakout continuation.
- In trending regime (chop < 38.2): Long on break above H3, short on break below L3.
- In ranging regime (chop > 61.8): Long on rejection at L3 (close > L3 after touch), short on rejection at H3 (close < H3 after touch).
- Exit: Opposite Camarilla level (H3 for longs, L3 for shorts) or regime shift.
- Discrete signal size: 0.25 to balance capture and drawdown.
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
    
    # Get 1d data for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3) from prior day OHLC
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # Note: We use prior day's OHLC, so shift by 1 to avoid look-ahead
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low)
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low)
    
    # Align 1d Camarilla levels to 12h (already completed prior day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume spike: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.5 * volume_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Chop regime: Chop > 61.8 = ranging, Chop < 38.2 = trending
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    # Chop = log10(sum(tr,14)/log10(14)) / log10(highest_high - lowest_low,14) * 100
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high_14 - lowest_low_14
    chop = np.where(
        (range_14 > 0) & (sum_tr_14 > 0),
        np.log10(sum_tr_14 / 14) / np.log10(range_14) * 100,
        50.0  # default to neutral if invalid
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_spike:
                if chop_val < 38.2:  # Trending regime: breakout
                    # Long on break above H3
                    if curr_close > h3:
                        signals[i] = 0.25
                        position = 1
                    # Short on break below L3
                    elif curr_close < l3:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: mean reversion at extremes
                    # Long on rejection at L3 (price touches L3 and closes above it)
                    if curr_low <= l3 and curr_close > l3:
                        signals[i] = 0.25
                        position = 1
                    # Short on rejection at H3 (price touches H3 and closes below it)
                    elif curr_high >= h3 and curr_close < h3:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR chop shifts to trending (exit ranging mean reversion)
            if curr_close < l3 or (chop_val < 38.2 and position == 1 and chop_val > 61.8):
                # Actually, simplify: exit on opposite level or regime shift to opposite extreme
                if curr_close < l3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR chop shifts to ranging (exit trending breakout)
            if curr_close > h3 or (chop_val > 61.8 and position == -1 and chop_val < 38.2):
                if curr_close > h3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    # Fix exit logic: simplify to clear rules
    # Re-implement exit logic clearly
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            if vol_spike:
                if chop_val < 38.2:  # Trending: breakout
                    if curr_close > h3:
                        signals[i] = 0.25
                        position = 1
                    elif curr_close < l3:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging: mean reversion
                    if curr_low <= l3 and curr_close > l3:
                        signals[i] = 0.25
                        position = 1
                    elif curr_high >= h3 and curr_close < h3:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price closes below L3 (failed support) OR chop shifts to strong trending (invalidates ranging mean reversion)
            if curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 (failed resistance) OR chop shifts to strong ranging (invalidates trending breakout)
            if curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dVolSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0