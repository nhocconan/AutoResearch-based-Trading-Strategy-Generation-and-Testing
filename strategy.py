#!/usr/bin/env python3
"""
Experiment #1880: 4h Donchian Breakout + HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts capture strong trending moves. Combined with 1d HMA(50) trend filter and volume confirmation (>1.5x average), this strategy enters only when price breaks out with institutional participation. ATR-based stoploss manages risk. Works in both bull and bear markets by following the 1d trend. Target: 75-200 total trades over 4 years (19-50/year) with position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1880_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d HMA(50) for trend direction
    def calculate_hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period == 0 or sqrt_period == 0:
            return arr.copy()
        wma1 = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
        wma2 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    hma_50_1d = calculate_hma(close_1d, 50)
    trend_1d = np.where(close_1d > hma_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for entry confirmation ===
    hma_21 = calculate_hma(close, 21)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for Donchian(20) and HMA(50) 1d
    
    for i in range(warmup, n):
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or reversal signal ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: price drops 2.5*ATR from highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below HMA(21) (trend weakening)
                elif price < hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: price rises 2.5*ATR from lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above HMA(21) (trend weakening)
                elif price > hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if 1d trend flips
                elif trend_1d_aligned[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_ok = vol_ratio[i] > 1.5
        
        if volume_ok:
            # Long entry: price breaks above Donchian upper + 1d uptrend + price > HMA(21)
            if trend_bias > 0 and price > highest_20[i] and price > hma_21[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower + 1d downtrend + price < HMA(21)
            elif trend_bias < 0 and price < lowest_20[i] and price < hma_21[i]:
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