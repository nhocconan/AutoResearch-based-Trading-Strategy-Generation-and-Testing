#!/usr/bin/env python3
"""
Experiment #5633: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with 12h HMA(21) trend direction capture high-probability trend continuation moves. 
The 12h HMA provides smoother trend filtering than EMA/SMA, reducing whipsaws in bear markets. 
Volume confirmation validates breakout strength. ATR-based trailing stop (2.5x ATR) manages risk. 
Discrete position sizing (0.25) minimizes fee churn. Target: 19-50 trades/year (75-200 total over 4 years).
Works in bull markets via breakout continuation and in bear markets via short breakdowns 
with volume confirmation, with HMA filter preventing counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5633_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan, dtype=np.float64)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    # Handle edge cases
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    
    # Pad to align lengths
    if len(wma_half) < len(series) or len(wma_full) < len(series):
        return np.full_like(series, np.nan, dtype=np.float64)
    
    # HMA = 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad to original length
    result = np.full_like(series, np.nan, dtype=np.float64)
    start_idx = len(series) - len(hma)
    if start_idx >= 0:
        result[start_idx:] = hma
    return result

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 12h data for HMA(21) trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        hma_12h = calculate_hma(df_12h['close'].values, 21)
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (structure break)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (structure break)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Trend filter: price above/below 12h HMA
        uptrend = price > hma_12h_aligned[i]
        downtrend = price < hma_12h_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volume
        long_setup = breakout_up and volume_confirmed and uptrend
        short_setup = breakout_down and volume_confirmed and downtrend
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals