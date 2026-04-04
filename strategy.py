#!/usr/bin/env python3
"""
Experiment #3487: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w data) and volume confirmation capture institutional flow. Weekly pivot provides structural bias (bullish/bearish) that works in both bull and bear markets by filtering breakouts to only those aligned with the weekly trend. Volume confirmation ensures breakouts have participation. Target: 75-150 total trades over 4 years (19-37/year). Position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3487_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from prior week's OHLC
    # Need to resample 1d to weekly - but we'll approximate using rolling window of 5 days
    # Actual weekly pivot: (Prior Week High + Prior Week Low + Prior Week Close) / 3
    lookback_week = 5  # 5 trading days ≈ 1 week
    prior_week_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
    prior_week_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
    prior_week_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).last().shift(1).values
    
    # Weekly pivot and support/resistance levels
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prior_week_low
    weekly_s1 = 2 * weekly_pivot - prior_week_high
    weekly_r2 = weekly_pivot + (prior_week_high - prior_week_low)
    weekly_s2 = weekly_pivot - (prior_week_high - prior_week_low)
    weekly_r3 = weekly_r2 + (prior_week_high - prior_week_low)
    weekly_s3 = weekly_s2 - (prior_week_high - prior_week_low)
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Determine weekly bias: bullish if price above weekly pivot, bearish if below
    weekly_bias = np.where(close_1d > weekly_pivot, 1, -1)  # 1 = bullish, -1 = bearish
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # === 6h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_6h = 20
    highest_high_6h = pd.Series(high).rolling(window=lookback_6h, min_periods=lookback_6h).max().values
    lowest_low_6h = pd.Series(low).rolling(window=lookback_6h, min_periods=lookback_6h).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = max(lookback_6h, 20, 14, lookback_week) + 5
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 6h Donchian channel (mean reversion)
                elif price <= highest_high_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 6h Donchian channel (mean reversion)
                elif price >= lowest_low_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            bias = weekly_bias_aligned[i]
            
            # Long entry: price breaks above 6h Donchian high with bullish weekly bias
            if (price > highest_high_6h[i] and 
                bias > 0):  # Weekly bullish bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian low with bearish weekly bias
            elif (price < lowest_low_6h[i] and 
                  bias < 0):  # Weekly bearish bias
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