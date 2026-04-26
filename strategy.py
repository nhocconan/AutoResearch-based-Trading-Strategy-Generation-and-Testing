#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d trend filter and chop regime filter.
Works in bull/bear via trend-following breakouts. Chop filter avoids whipsaws in ranging markets.
Target: 12-37 trades/year per symbol (~50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d choppiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels on 12h using previous 12h bar
    # Camarilla: based on previous day's (here: previous 12h bar) range
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous bar values
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R1, S1, R3, S3
    range_12h = prev_high - prev_low
    camarilla_r1 = prev_close + range_12h * 1.1 / 12
    camarilla_s1 = prev_close - range_12h * 1.1 / 12
    camarilla_r3 = prev_close + range_12h * 1.1 / 4
    camarilla_s3 = prev_close - range_12h * 1.1 / 4
    
    # Align 12h indicators to 12h timeframe (they're already aligned)
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    camarilla_r3_aligned = camarilla_r3
    camarilla_s3_aligned = camarilla_s3
    
    # Align 1d indicators
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and chop (14)
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        # We want trending regime for breakouts
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + trending regime
            if close[i] > camarilla_r1_aligned[i] and price_above_ema and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + trending regime
            elif close[i] < camarilla_s1_aligned[i] and price_below_ema and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend changes OR chop becomes ranging
            if (close[i] < camarilla_s1_aligned[i] or 
                not price_above_ema or 
                not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend changes OR chop becomes ranging
            if (close[i] > camarilla_r1_aligned[i] or 
                not price_below_ema or 
                not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter"
timeframe = "12h"
leverage = 1.0