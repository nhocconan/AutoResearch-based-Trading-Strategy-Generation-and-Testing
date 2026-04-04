#!/usr/bin/env python3
"""
Experiment #2308: 12h Donchian Breakout + 1w/1d Trend + Volume Spike
HYPOTHESIS: 12h Donchian(20) breakouts with 1d and 1w EMA(50) trend alignment and volume confirmation capture strong directional moves while avoiding chop. Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band). Discrete position sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2308_12h_donchian20_1d1w_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend and prior OHLC for Donchian calculation ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for EMA trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20) from 1d data, ATR(14), Volume MA(20) ===
    # Calculate Donchian channels using prior 20 days (shifted by 1 to avoid look-ahead)
    if len(high_1d) >= 20:
        # Rolling max/min of high/low over past 20 days, shifted by 1 for prior completed period
        high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
        # Shift by 1 to get prior 20-day channel (avoid look-ahead)
        high_20_prev = np.roll(high_20, 1)
        low_20_prev = np.roll(low_20, 1)
        high_20_prev[0] = np.nan
        low_20_prev[0] = np.nan
    else:
        high_20_prev = np.array([])
        low_20_prev = np.array([])
    
    # Align prior 20-day Donchian to 12h timeframe
    high_20_prev_aligned = align_htf_to_ltf(prices, df_1d, high_20_prev)
    low_20_prev_aligned = align_htf_to_ltf(prices, df_1d, low_20_prev)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]) or
            np.isnan(high_20_prev_aligned[i]) or np.isnan(low_20_prev_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price breaks below lower Donchian (trend reversal)
                elif price < low_20_prev_aligned[i]:
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
                # Exit if price breaks above upper Donchian (trend reversal)
                elif price > high_20_prev_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 1d and 1w trend alignment for bias filter
        trend_bias_1d = trend_1d_aligned[i]
        trend_bias_1w = trend_1w_aligned[i]
        
        # Only trade when both timeframes agree
        if trend_bias_1d == trend_bias_1w:
            trend_bias = trend_bias_1d  # Either 1 or -1
        else:
            trend_bias = 0  # No clear trend, skip
        
        # Volume confirmation: require volume spike (> 2.0x average - strict to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper Donchian with uptrend on both timeframes
            if trend_bias > 0 and price > high_20_prev_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian with downtrend on both timeframes
            elif trend_bias < 0 and price < low_20_prev_aligned[i]:
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