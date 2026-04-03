#!/usr/bin/env python3
"""
Experiment #1927: 6h Donchian Breakout + 1d Weekly Pivot + Volume Confirmation
HYPOTHESIS: Donchian channel breakouts on 6h timeframe, when aligned with 1d weekly pivot 
direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) and 
confirmed by volume spike (>2x average), provide high-probability entries. 
Weekly pivot acts as institutional reference point. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1927_6h_donchian_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d OHLC for weekly pivot (using prior week's data)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll approximate using 5-period rolling on 1d data (5 trading days = 1 week)
    if len(high_1d) >= 5:
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values  # approx
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full_like(close_1d, np.nan)
    
    # Align weekly pivot to 6h timeframe (shifted by 1 for completed bars only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 1d trend filter: EMA(20)
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_20_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Donchian Channel (20) and Volume MA(20) ===
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
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
    
    warmup = 50  # sufficient for Donchian(20), EMA(20), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price returns to Donchian midpoint (mean reversion)
                dc_mid = (dc_high[i] + dc_low[i]) / 2.0
                if price <= dc_mid:
                    exit_signal = True
                # Exit if price touches Donchian lower band (stop loss)
                elif price <= dc_low[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to Donchian midpoint
                dc_mid = (dc_high[i] + dc_low[i]) / 2.0
                if price >= dc_mid:
                    exit_signal = True
                # Exit if price touches Donchian upper band (stop loss)
                elif price >= dc_high[i]:
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
        # Require volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND price > weekly pivot (bullish bias)
            if price > dc_high[i] and price > weekly_pivot_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND price < weekly pivot (bearish bias)
            elif price < dc_low[i] and price < weekly_pivot_aligned[i]:
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