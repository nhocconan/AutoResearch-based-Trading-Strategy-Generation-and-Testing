#!/usr/bin/env python3
"""
Experiment #3993: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 12h HMA(21) trend capture sustained moves with less whipsaw.
Volume > 1.5x MA(20) confirms breakout strength. ATR(20) trailing stop (2.5x) manages risk.
Discrete sizing (0.25) reduces fee drag. Target: 75-200 trades over 4 years (19-50/year).
Works in bull/bear via 12h HMA as trend filter (avoids counter-trend trades).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3993_4h_donchian20_12h_hma_vol_v1"
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
    if len(df_12h) >= 21:
        # Calculate HMA(21) on 12h close
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = pd.Series(df_12h['close'].values).rolling(window=half_len, min_periods=half_len).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
        ).values
        wma_full = pd.Series(df_12h['close'].values).rolling(window=21, min_periods=21).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
        ).values
        
        hma_raw = 2 * wma_half - wma_full
        hma_12h = pd.Series(hma_raw).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
        ).values
        
        # Align to LTF (4h) with proper shift(1) for completed bars only
        hma_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(lookback_dc + 1, 20, 20, 21 + 10)  # DC lookback, vol MA, ATR, HMA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
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
            # Determine trend alignment from 12h HMA
            # Bullish bias: price above 12h HMA
            # Bearish bias: price below 12h HMA
            bullish_bias = price > hma_aligned[i]
            bearish_bias = price < hma_aligned[i]
            
            # Breakout conditions
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            if bullish_bias and breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif bearish_bias and breakout_down:
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