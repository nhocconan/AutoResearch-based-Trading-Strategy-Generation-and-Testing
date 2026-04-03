#!/usr/bin/env python3
"""
Experiment #1866: 4h Donchian(20) breakout + 1d HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Combined with 1d HMA(21) trend filter and volume confirmation (>1.5x average), this strategy avoids false breakouts in choppy markets. Works in both bull and bear markets by following the 1d trend direction. Target: 100-200 total trades over 4 years (25-50/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1866_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d HMA(21) for trend direction
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_val = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
        return hma_val.values
    
    hma_21_1d = hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_21_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel(20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for Donchian(20) and ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian high + 1d uptrend
            if trend_bias > 0 and price > donchian_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low + 1d downtrend
            elif trend_bias < 0 and price < donchian_low[i]:
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