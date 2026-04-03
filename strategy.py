#!/usr/bin/env python3
"""
Experiment #1934: 1h Donchian Breakout + 4h/1d Trend + Volume Spike
HYPOTHESIS: 1h Donchian(20) breakouts aligned with 4h/1d trend filters and volume spikes capture institutional momentum while minimizing whipsaws. Uses 4h EMA(50) and 1d EMA(200) for trend alignment, volume > 2x 20-period average for confirmation. Session filter (08-20 UTC) reduces off-hours noise. Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1934_1h_donchian_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA(50) trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_50_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for EMA(200) trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_200_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    max_favorable_price = 0.0
    
    warmup = 200  # sufficient for 1d EMA(200)
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Update max favorable price for trailing stop
            if position_side > 0:  # Long
                max_favorable_price = max(max_favorable_price, price)
            else:  # Short
                max_favorable_price = min(max_favorable_price, price)
            
            # Exit conditions
            exit_signal = False
            
            # 1. Trend reversal on either timeframe
            if (position_side > 0 and (trend_4h_aligned[i] < 0 or trend_1d_aligned[i] < 0)) or \
               (position_side < 0 and (trend_4h_aligned[i] > 0 or trend_1d_aligned[i] > 0)):
                exit_signal = True
            
            # 2. Price retracement of 50% from peak adverse excursion
            elif position_side > 0:  # Long
                adverse_excursion = entry_price - price
                if adverse_excursion > 0 and bars_since_entry >= 5:  # Allow some room initially
                    max_adverse = entry_price - np.min(low[i-bars_since_entry:i+1])
                    if max_adverse > 0 and adverse_excursion >= 0.5 * max_adverse:
                        exit_signal = True
            else:  # Short
                adverse_excursion = price - entry_price
                if adverse_excursion > 0 and bars_since_entry >= 5:
                    max_adverse = np.max(high[i-bars_since_entry:i+1]) - entry_price
                    if max_adverse > 0 and adverse_excursion >= 0.5 * max_adverse:
                        exit_signal = True
            
            # 3. Time-based exit: max 24 hours (24 bars on 1h)
            elif bars_since_entry >= 24:
                exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                max_favorable_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 4h and 1d trend alignment for bias filter
        trend_bias_4h = trend_4h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias_4h == trend_bias_1d:
            # Long entry: price breaks above Donchian high AND both trends up
            if trend_bias_4h > 0 and trend_bias_1d > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                max_favorable_price = price
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND both trends down
            elif trend_bias_4h < 0 and trend_bias_1d < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                max_favorable_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals