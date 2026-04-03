#!/usr/bin/env python3
"""
Experiment #1868: 12h Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: 12h Donchian breakouts capture medium-term momentum. 1w HMA(21) filters for primary trend alignment. Volume spike (>2x MA20) confirms institutional participation. ATR-based stoploss manages risk. Works in bull/bear by following 1w trend. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1868_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w HMA(21) for trend direction
    def calculate_hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, min_periods=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean()
        raw_hma = 2 * wma2 - wma1
        hma = pd.Series(raw_hma).ewm(span=sqrt, min_periods=sqrt, adjust=False).mean()
        return hma.values
    
    hma_21_1w = calculate_hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian Channel(20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or trend reversal ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5 * ATR below entry
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
                # Exit if 1w trend flips bearish
                elif trend_1w_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2.5 * ATR above entry
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
                # Exit if 1w trend flips bullish
                elif trend_1w_aligned[i] > 0:
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
        # Require 1w trend alignment for bias
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Donchian breakout in direction of 1w trend
            if trend_bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif trend_bias < 0 and price < lowest_20[i]:
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