#!/usr/bin/env python3
"""
Experiment #4867: 6h Donchian(20) Breakout + 1d Camarilla Pivot + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts in direction of 1d Camarilla pivot bias (price > pivot = bullish bias, price < pivot = bearish bias) with volume confirmation (>1.5x average) capture momentum moves while avoiding false breakouts in ranging markets. Uses ATR(14) trailing stop (2.0x) for risk management. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend) by using pivot as dynamic bias filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4867_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, R4, S3, S4, Pivot) ===
    if len(df_1d) >= 1:
        # Calculate pivot and levels from previous day's OHLC
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Pivot = (High + Low + Close) / 3
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        
        # Camarilla levels
        range_1d = high_1d - low_1d
        r3_1d = close_1d + range_1d * 1.1 / 4.0
        r4_1d = close_1d + range_1d * 1.1 / 2.0
        s3_1d = close_1d - range_1d * 1.1 / 4.0
        s4_1d = close_1d - range_1d * 1.1 / 2.0
        
        # Bias: price > pivot = bullish bias, price < pivot = bearish bias
        bias_bullish = pivot_1d  # we'll compare price > pivot for bullish
        bias_bearish = pivot_1d  # we'll compare price < pivot for bearish
    else:
        # Not enough data
        pivot_1d = np.array([])
        r3_1d = np.array([])
        r4_1d = np.array([])
        s3_1d = np.array([])
        s4_1d = np.array([])
        bias_bullish = np.array([])
        bias_bearish = np.array([])
    
    # Align HTF Camarilla levels to 6h timeframe
    if len(pivot_1d) > 0:
        pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
        r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
        s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
        s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    else:
        pivot_1d_aligned = np.full(n, np.nan)
        r3_1d_aligned = np.full(n, np.nan)
        r4_1d_aligned = np.full(n, np.nan)
        s3_1d_aligned = np.full(n, np.nan)
        s4_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with Camarilla pivot bias
        # Bullish bias: price > pivot -> look for longs on breakout
        # Bearish bias: price < pivot -> look for shorts on breakdown
        bullish_bias = price > pivot_1d_aligned[i]
        bearish_bias = price < pivot_1d_aligned[i]
        
        breakout_long = (price >= high_roll[i]) and bullish_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and bearish_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals