#!/usr/bin/env python3
"""
Experiment #3485: 12h Donchian Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: 12h Donchian(20) breakouts with volume confirmation and 1d HMA trend filter capture medium-term momentum in both bull and bear markets. The 12h timeframe reduces trade frequency to avoid fee drag while HMA provides smooth trend detection. Volume spike confirms breakout validity. Works in bull (trend continuation) and bear (mean reversion from extremes via channel exit).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3485_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on 1d: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        if half == 0 or sqrt == 0:
            return arr.copy()
        wma_half = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma_half - wma_full
        hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
        return hma
    
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Indicators: Donchian channels (20-period) for entry/exit ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 12h Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 12h Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # 1d HMA trend filter: only long above HMA, short below HMA
            price_vs_hma = price - hma_1d_aligned[i]
            
            # Long entry: price breaks above 12h Donchian high with bullish 1d HMA trend
            if (price > highest_high[i] and 
                price_vs_hma > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 12h Donchian low with bearish 1d HMA trend
            elif (price < lowest_low[i] and 
                  price_vs_hma < 0):
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