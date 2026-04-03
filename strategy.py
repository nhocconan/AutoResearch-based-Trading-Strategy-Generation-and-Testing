#!/usr/bin/env python3
"""
Experiment #2014: 1h Donchian(20) breakout + 4h/1d EMA trend + volume confirmation
HYPOTHESIS: 1h Donchian breakouts with volume spike capture short-term momentum,
filtered by 4h/1d EMA trend alignment to avoid counter-trend whipsaws.
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
Uses 4h/1d for signal direction, 1h only for entry timing precision.
Session filter (08-20 UTC) reduces noise trades outside active market hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2014_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # EMA(21) on 4h
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
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
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
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
        # Trend bias: require both 4h and 1d EMA alignment
        trend_4h_bias = 1 if close[i] > ema_4h_aligned[i] else -1
        trend_1d_bias = 1 if close[i] > ema_1d_aligned[i] else -1
        # Only trade when both timeframes agree
        trend_bias = trend_4h_bias if trend_4h_bias == trend_1d_bias else 0
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper Donchian AND trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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