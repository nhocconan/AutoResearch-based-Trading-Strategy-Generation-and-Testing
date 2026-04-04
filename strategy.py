#!/usr/bin/env python3
"""
Experiment #4673: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: 4h price breaking Donchian(20) channels with volume confirmation and 12h HMA trend filter captures momentum in both bull and bear markets. The 12h HMA acts as a regime filter to avoid counter-trend trades, while volume confirms breakout strength. Target: 19-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4673_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: HMA(21) for trend filter ===
    if len(df_12h) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = len(df_12h) // 2
        sqrt_len = int(np.sqrt(len(df_12h)))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len) if half_len > 0 else np.array([])
        wma_full = wma(close_12h, len(close_12h))
        if len(wma_half) > 0 and len(wma_full) > 0:
            hma_raw = 2 * wma_half[-len(wma_full):] - wma_full
            hma = wma(hma_raw, sqrt_len) if sqrt_len > 0 else hma_raw
            # Pad to original length
            hma_padded = np.full(len(close_12h), np.nan)
            start_idx = len(close_12h) - len(hma)
            if start_idx >= 0:
                hma_padded[start_idx:] = hma
            hma_12h = hma_padded
        else:
            hma_12h = np.full(len(close_12h), np.nan)
    else:
        hma_12h = np.full(len(df_12h), np.nan)
    
    # Align HTF indicators to 4h timeframe
    if len(hma_12h) > 0:
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) breakout ===
    donchian_len = 20
    if n >= donchian_len:
        donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().shift(1).values
        donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().shift(1).values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = max(donchian_len, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation for breakouts (>1.5x)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Trend filter: 12h HMA direction
        hma_trend_up = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_trend_down = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Breakout conditions: price breaks Donchian high/low with volume and trend confirmation
        breakout_long = price > donchian_high[i] and vol_breakout and hma_trend_up
        breakout_short = price < donchian_low[i] and vol_breakout and hma_trend_down
        
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals