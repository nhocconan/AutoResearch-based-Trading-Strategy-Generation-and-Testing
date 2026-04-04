#!/usr/bin/env python3
"""
Experiment #4633: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: 4h price breaking Donchian(20) channels from prior 20 periods with 12h HMA trend filter and volume confirmation (>1.3x avg) captures strong momentum breakouts while avoiding counter-trend whipsaws. Uses discrete sizing (0.25) and ATR trailing stop (2.0x) to manage risk. Target: 19-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4633_4h_donchian20_12h_hma_vol_v1"
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
    if len(df_12h) >= 21:
        hma_12h = calculate_hma(df_12h['close'].values, 21)
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) from prior 20 periods ===
    if n >= 20:
        # Use prior 20 periods' high/low (shifted by 1 to avoid look-ahead)
        ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 periods high
        pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 periods low
        
        # Rolling max/min of prior 20 periods
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
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
    
    warmup = max(20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation for breakouts (>1.3x)
        vol_breakout = vol_ratio[i] > 1.3
        
        # Trend filter: 12h HMA direction
        # For long: price above 12h HMA (uptrend)
        # For short: price below 12h HMA (downtrend)
        trend_long = price > hma_12h_aligned[i]
        trend_short = price < hma_12h_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume and trend confirmation
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

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(values)
    for i in range(half_period, len(values)):
        wma_half[i] = np.sum(values[i-half_period+1:i+1] * np.arange(1, half_period+1)) / (half_period * (half_period + 1) / 2)
    
    # WMA of full period
    wma_full = np.zeros_like(values)
    for i in range(period, len(values)):
        wma_full[i] = np.sum(values[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = np.zeros_like(values)
    for i in range(sqrt_period, len(values)):
        hma[i] = np.sum(raw_hma[i-sqrt_period+1:i+1] * np.arange(1, sqrt_period+1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma