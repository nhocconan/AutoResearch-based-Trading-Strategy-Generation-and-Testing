#!/usr/bin/env python3
"""
Experiment #1863: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. HMA(21) filters for trend direction, volume confirmation (>1.5x average) ensures institutional participation, and ATR(14) stoploss manages risk. Works in both bull and bear markets by following the 12h trend. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1863_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h HMA(21) for trend direction
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
        return hma_vals.values
    
    hma_21_12h = hma(close_12h, 21)
    trend_12h = np.where(close_12h > hma_21_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4h Indicators: Donchian(20) channels ===
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss and position sizing ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = 50  # sufficient for Donchian(20) and ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or opposite breakout ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2 * ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    exit_signal = True
                # Exit on Donchian lower break (failed breakout)
                elif price < dc_lower[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2 * ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    exit_signal = True
                # Exit on Donchian upper break (failed breakout)
                elif price > dc_upper[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment for bias
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Long entry: price breaks above Donchian upper with bullish 12h trend
            if trend_bias > 0 and price > dc_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower with bearish 12h trend
            elif trend_bias < 0 and price < dc_lower[i]:
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