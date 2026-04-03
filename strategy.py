#!/usr/bin/env python3
"""
Experiment #2087: 6h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with multi-timeframe pivot alignment capture institutional 
order flow while avoiding overtrading. Uses 1d/1w Camarilla pivots for trend filter and 
volume confirmation for signal quality. Designed for 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2087_6h_donchian20_1d_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivots
    # Pivot = (H + L + C) / 3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    r2_1d = close_1d + range_1d * 1.1 / 6.0
    s2_1d = close_1d - range_1d * 1.1 / 6.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Trend bias from 1d: 1 if close > R3 (bullish bias), -1 if close < S3 (bearish bias), 0 otherwise
    trend_bias_1d = np.where(close_1d > r3_1d, 1, np.where(close_1d < s3_1d, -1, 0))
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Breakout bias from 1d: 1 if close > R4 (strong bullish), -1 if close < S4 (strong bearish), 0 otherwise
    breakout_bias_1d = np.where(close_1d > r4_1d, 1, np.where(close_1d < s4_1d, -1, 0))
    breakout_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, breakout_bias_1d)
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivots (same formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    
    # Weekly trend bias: 1 if close > R3 (bullish), -1 if close < S3 (bearish), 0 otherwise
    weekly_bias = np.where(close_1w > r3_1w, 1, np.where(close_1w < s3_1w, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_bias_1d_aligned[i]) or np.isnan(breakout_bias_1d_aligned[i]) or
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine combined bias from timeframes
            # Long bias: 1d trend up OR 1d breakout up AND weekly up
            long_bias = (trend_bias_1d_aligned[i] > 0 or breakout_bias_1d_aligned[i] > 0) and weekly_bias_aligned[i] >= 0
            # Short bias: 1d trend down OR 1d breakout down AND weekly down
            short_bias = (trend_bias_1d_aligned[i] < 0 or breakout_bias_1d_aligned[i] < 0) and weekly_bias_aligned[i] <= 0
            
            # Long entry: price breaks above upper Donchian AND long bias
            if long_bias and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND short bias
            elif short_bias and price < donchian_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals