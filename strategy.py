#!/usr/bin/env python3
"""
Experiment #4011: 6h Donchian(20) breakout + daily pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with daily Camarilla pivot S3/R3 levels capture high-probability mean-reversion and breakout trades in both bull and bear markets. Volume > 1.5x MA(20) confirms participation. Discrete sizing (0.25) and ATR(20) trailing stop (2.0x) control risk. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4011_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for daily Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate daily Camarilla pivots from previous day
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_ = prev_high - prev_low
        # Camarilla levels: S3 = pivot - 1.1*range_/2, R3 = pivot + 1.1*range_/2
        s3 = pivot - 1.1 * range_ / 2.0
        r3 = pivot + 1.1 * range_ / 2.0
        # Align to 6h timeframe (shifted by 1 for completed daily bar)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    else:
        s3_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
    
    warmup = max(lookback_dc + 1, 20 + 10, 20 + 10, 2 + 5)  # DC lookback, vol MA, ATR buffer, HTF buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i])):
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Mean reversion at S3/R3: long near S3, short near R3
            # Breakout beyond S3/R3: short below S3, long above R3 (continuation)
            near_s3 = abs(price - s3_aligned[i]) < 0.5 * atr[i]  # within 0.5 ATR of S3
            near_r3 = abs(price - r3_aligned[i]) < 0.5 * atr[i]  # within 0.5 ATR of R3
            breakout_below_s3 = price < s3_aligned[i]
            breakout_above_r3 = price > r3_aligned[i]
            
            # Long: mean reversion at S3 OR breakout above R3 (with Donchian confirmation)
            if (near_s3 or breakout_above_r3) and price > highest_high[i-1]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: mean reversion at R3 OR breakout below S3 (with Donchian confirmation)
            elif (near_r3 or breakout_below_s3) and price < lowest_low[i-1]:
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