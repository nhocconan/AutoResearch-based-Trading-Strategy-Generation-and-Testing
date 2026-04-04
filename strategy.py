#!/usr/bin/env python3
"""
Experiment #4670: 1d Donchian(20) Breakout + 1w HMA Trend Filter + Volume Confirmation
HYPOTHESIS: Daily price breaking Donchian(20) channels (from prior 20 daily bars) with volume confirmation and weekly HMA trend filter captures strong momentum. Weekly HMA avoids counter-trend trades in bear markets. Target: 7-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4670_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Donchian(20) from prior 20 days ===
    if len(close) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === 1w Indicators: HMA(21) for trend filter ===
    if len(df_1w) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = len(df_1w) // 2
        sqrt_len = int(np.sqrt(len(df_1w)))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        wma_half = wma(close_1w, half_len) if half_len > 0 else np.full_like(close_1w, np.nan)
        wma_full = wma(close_1w, len(close_1w))
        wma_2x_sub = 2 * wma_half[-len(wma_full):] - wma_full
        hma = wma(wma_2x_sub, sqrt_len) if sqrt_len > 0 else np.full_like(wma_2x_sub, np.nan)
        
        # Pad to match df_1w length
        hma_padded = np.full(len(close_1w), np.nan)
        hma_padded[-len(hma):] = hma
        hma_1w = hma_padded
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # Align HTF indicators to 1d timeframe
    if len(hma_1w) > 0:
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 14)  # Donchian, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: price above/below weekly HMA
        trend_long = price > hma_1w_aligned[i]
        trend_short = price < hma_1w_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation and trend filter
        breakout_long = price > donchian_high[i] and vol_breakout and trend_long
        breakout_short = price < donchian_low[i] and vol_breakout and trend_short
        
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