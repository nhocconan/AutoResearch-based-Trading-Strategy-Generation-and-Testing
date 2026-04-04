#!/usr/bin/env python3
"""
Experiment #4013: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 12h Hull Moving Average trend capture high-probability trades in both bull and bear markets. Volume > 1.5x MA(20) confirms participation. ATR(20) trailing stop (2.0x) controls risk. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4013_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 9:
        # Calculate HMA(9) on 12h close
        close_12h = df_12h['close'].values
        # WMA function
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        half_len = len(close_12h) // 2
        sqrt_len = int(np.sqrt(len(close_12h)))
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, len(close_12h))
        wma_sqrt = wma(close_12h, sqrt_len)
        # HMA = 2*WMA(half) - WMA(full), then WMA of that with sqrt period
        hma_raw = 2 * wma_half - wma_full
        hma_12h = wma(hma_raw, sqrt_len)
        # Pad to original length
        hma_12h_padded = np.full(len(close_12h), np.nan)
        hma_12h_padded[-len(hma_12h):] = hma_12h
        # Align to 4h timeframe (shifted by 1 for completed 12h bar)
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10, 20 + 10, 9 + 5)  # DC lookback, vol MA, ATR buffer, HTF buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Trend alignment: price > 12h HMA = uptrend bias, price < 12h HMA = downtrend bias
            uptrend_bias = price > hma_12h_aligned[i]
            downtrend_bias = price < hma_12h_aligned[i]
            
            # Long: Donchian breakout above resistance in uptrend OR mean reversion at support in downtrend
            if (uptrend_bias and price > highest_high[i-1]) or (downtrend_bias and price < lowest_low[i-1]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: Donchian breakout below support in downtrend OR mean reversion at resistance in uptrend
            elif (downtrend_bias and price < lowest_low[i-1]) or (uptrend_bias and price > highest_high[i-1]):
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