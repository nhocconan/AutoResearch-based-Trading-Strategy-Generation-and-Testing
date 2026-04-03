#!/usr/bin/env python3
"""
Experiment #1981: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts capture medium-term momentum, filtered by 1d EMA trend direction and volume spikes to avoid false breakouts. Works in bull/bear by following higher timeframe trend. Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1981_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donch_high[i] = np.max(high[i-lookback:i])
        donch_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x 20-period average
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
    bars_since_entry = 0
    
    warmup = max(50, lookback)  # sufficient for EMA(50) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: reverse Donchian breakout or ATR-based stoploss
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian Lower (reverse breakout)
                if price < donch_low[i]:
                    exit_signal = True
                # Optional: time-based exit after 10 bars to prevent stagnation
                elif bars_since_entry >= 10:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian Upper (reverse breakout)
                if price > donch_high[i]:
                    exit_signal = True
                # Optional: time-based exit after 10 bars
                elif bars_since_entry >= 10:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation and 1d trend alignment
        volume_spike = vol_ratio[i] > 1.5
        trend_bias = trend_1d_aligned[i]
        
        if volume_spike:
            # Long entry: price breaks above Donchian Upper AND 1d uptrend
            if trend_bias > 0 and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian Lower AND 1d downtrend
            elif trend_bias < 0 and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals