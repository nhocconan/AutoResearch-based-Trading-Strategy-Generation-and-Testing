#!/usr/bin/env python3
"""
Experiment #3247: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts capture medium-term trends with controlled trade frequency.
Weekly pivot direction (from 1d HTF) filters breakouts to align with higher timeframe momentum.
Volume confirmation (>2.0x 20-period average) ensures breakout strength.
ATR-based trailing stop (2.5x) manages risk. Position size 0.25.
Designed to work in both bull (trend continuation) and bear (mean reversion from extremes) markets
by using price channels and volatility filters with weekly structural bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3247_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # For each 1d bar, we need the prior week's high, low, close
    # We'll approximate by using rolling weekly lookback on daily data
    lookback_week = 5  # 5 trading days approx 1 week
    weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).mean().shift(1).values
    
    # Weekly pivot point and support/resistance levels
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pp - weekly_low
    weekly_s1 = 2 * weekly_pp - weekly_high
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pp - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pp)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Determine weekly bias: price above/below weekly pivot
    weekly_bias = np.where(close_1d > weekly_pp, 1, -1)  # 1=bullish, -1=bearish
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
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
    
    warmup = max(50, lookback, 20, 14, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_bias_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
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
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Weekly pivot bias filter: only long in bullish bias, short in bearish bias
            bias = weekly_bias_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish weekly bias
            if price > highest_high[i] and bias > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish weekly bias
            elif price < lowest_low[i] and bias < 0:
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