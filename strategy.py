#!/usr/bin/env python3
"""
Experiment #827: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts capture momentum, filtered by weekly pivot bias (price above/below weekly pivot from 1d data) and volume spike (>2.0x average). 
Weekly pivot provides structural bias: above pivot = bullish bias (favor longs), below pivot = bearish bias (favor shorts). 
Works in bull/bear markets: in bull trends, pivot bias filters for quality longs; in bear markets, pivot bias filters for quality shorts. 
Uses discrete position sizing (0.25). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_827_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # Calculate weekly pivot points from prior week's daily OHLC
    # Using simplified weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    # We'll approximate with rolling window of 5 trading days (1 week)
    def calculate_weekly_pivot(high_arr, low_arr, close_arr, window=5):
        # Rolling weekly high, low, close
        weekly_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        weekly_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        weekly_close = pd.Series(close_arr).rolling(window=window, min_periods=window).last().values
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        return pivot
    
    weekly_pivot_1d = calculate_weekly_pivot(high_1d, low_1d, close_1d, window=5)
    # Pivot bias: 1 = price above weekly pivot (bullish), -1 = price below weekly pivot (bearish), 0 = at pivot
    pivot_bias_1d = np.zeros_like(weekly_pivot_1d)
    # Use prior week's pivot for current week's bias (avoid look-ahead)
    pivot_bias_1d[5:] = np.where(close_1d[5:] > weekly_pivot_1d[:-5], 1,
                                 np.where(close_1d[5:] < weekly_pivot_1d[:-5], -1, 0))
    # Align bias to 6h timeframe
    pivot_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias_1d)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(pivot_bias_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND pivot bias bullish (1)
            if price > upper_20[i] and pivot_bias_1d_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND pivot bias bearish (-1)
            elif price < lower_20[i] and pivot_bias_1d_aligned[i] < 0:
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