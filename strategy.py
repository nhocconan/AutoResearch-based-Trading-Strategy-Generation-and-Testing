#!/usr/bin/env python3
"""
Experiment #700: 4h Donchian(20) breakout + HMA(21) trend + Volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture institutional momentum, filtered by HMA trend direction and volume confirmation. 
Uses 1d HTF for regime filter (only trade in direction of 1d HMA(50)). Discrete position sizing (0.25) minimizes fee churn. 
Works in bull/bear markets via 1d trend filter: long only when price > 1d HMA(50), short only when price < 1d HMA(50). 
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_700_4h_donchian20_hma21_1d_hma50_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(50) for regime filter
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=np.float64)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
        return hma_vals
    
    hma_1d = hma(close_1d, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: HMA(21) for trend filter ===
    hma_4h = hma(close, 21)
    
    # === 4h Indicators: Volume MA(20) for spike confirmation ===
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_4h[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~24h on 4h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Get trend filters
            hma_trend_4h = hma_4h[i]  # 4h HMA(21) direction
            regime_1d = hma_1d_aligned[i]  # 1d HMA(50) regime
            
            # Long: price breaks above Donchian upper + above 4h HMA + price > 1d HMA
            if (price > highest_high[i] and 
                price > hma_trend_4h and 
                price > regime_1d):
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower + below 4h HMA + price < 1d HMA
            elif (price < lowest_low[i] and 
                  price < hma_trend_4h and 
                  price < regime_1d):
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals