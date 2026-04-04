#!/usr/bin/env python3
"""
Experiment #3987: 6h Donchian(20) breakout + 1d/1w weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d/1w weekly pivot structure (pivot, R1, S1) capture multi-swing trades. 
Weekly pivot > 1d pivot = bullish bias (long breakouts/fade longs); weekly pivot < 1d pivot = bearish bias (short breakdowns/fade shorts). 
Volume > 2.0x MA(30) confirms strength. ATR(20) trailing stop (2.0x) manages risk. Discrete sizing (0.25) minimizes fee drag. 
Target: 75-150 trades over 4 years (19-38/year). Works in bull/bear via pivot hierarchy as institutional reference.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3987_6h_donchian20_1d1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for weekly pivot bias ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = typical_price_1w
    weekly_r1 = 2.0 * weekly_pivot - low_1w
    weekly_s1 = 2.0 * weekly_pivot - high_1w
    
    # Daily pivot for reference levels
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_pivot = typical_price_1d
    daily_r1 = 2.0 * daily_pivot - low_1d
    daily_s1 = 2.0 * daily_pivot - high_1d
    
    # Pivot bias: weekly pivot > daily pivot = bullish bias; < = bearish bias
    pivot_bias = weekly_pivot - daily_pivot  # >0 bullish, <0 bearish
    
    # Align to 6h timeframe (shift by 1 for completed HTF bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1w, pivot_bias)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(30) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[30:] = volume[30:] / vol_ma[30:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 30, 20)  # DC lookback, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(pivot_bias_aligned[i]) or
            np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below weekly S1 (strong support)
                elif price < weekly_s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above weekly R1 (strong resistance)
                elif price > weekly_r1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            bullish_bias = pivot_bias_aligned[i-1] > 0
            bearish_bias = pivot_bias_aligned[i-1] < 0
            
            # Long logic: 
            # - Bullish bias: breakout above daily R1 OR fade from weekly S1 with bullish bias
            # - Bearish bias: only fade from weekly S1 (counter-trend bounce in bear)
            long_breakout = bullish_bias and price > daily_r1_aligned[i-1]
            long_fade = price < weekly_s1_aligned[i-1] and price > weekly_s1_aligned[i-1] * 0.995  # Near weekly S1 bounce
            
            # Short logic:
            # - Bearish bias: breakdown below daily S1 OR fade from weekly R1 with bearish bias
            # - Bullish bias: only fade from weekly R1 (counter-trend rejection in bull)
            short_breakout = bearish_bias and price < daily_s1_aligned[i-1]
            short_fade = price > weekly_r1_aligned[i-1] and price < weekly_r1_aligned[i-1] * 1.005  # Near weekly R1 rejection
            
            if (long_breakout or long_fade) and not (short_breakout or short_fade):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif (short_breakout or short_fade) and not (long_breakout or long_fade):
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