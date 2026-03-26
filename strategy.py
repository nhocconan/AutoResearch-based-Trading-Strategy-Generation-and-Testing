#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Choppiness Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 4h capture institutional moves.
Choppiness Index filters out ranging markets (>50 = skip, <50 = trade).
1d HMA provides trend bias. Volume spike confirms conviction.
Works in bull (long breakouts) and bear (short breakdowns to 1d HMA).

TIMEFRAME: 4h primary
HTF: 1d for trend bias
TARGET: 75-150 total trades over 4 years (~20-40/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    Values between are neutral
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of true range over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    # ATR trailing stop parameters
    ATR_MULT = 2.5
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK: Choppiness Index ===
        # < 38.2 = trending (good for breakouts)
        # > 61.8 = choppy (skip breakouts, could mean revert)
        chop_val = chop[i]
        is_trending = chop_val < 50.0  # Liberal threshold to get more trades
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout up: close exceeds upper band
        price_above_upper = close[i] > donch_upper[i]
        # Breakout down: close falls below lower band
        price_below_lower = close[i] < donch_lower[i]
        
        # === ATR TRAILING STOP ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - ATR_MULT * entry_atr
            if low[i] < stop_price:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + ATR_MULT * entry_atr
            if high[i] > stop_price:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout above upper band + volume + bullish trend + trending regime
            if price_above_upper and vol_spike and trend_bullish and is_trending:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakdown below lower band + volume + bearish trend + trending regime
            if price_below_lower and vol_spike and trend_bearish and is_trending:
                desired_signal = -SIZE
        
        # === EXIT: Opposite band or stop ===
        if in_position and position_side > 0:
            # Long exit: price falls below lower band (mean revert signal)
            if price_below_lower:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Short exit: price rises above upper band
            if price_above_upper:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - ATR_MULT * entry_atr
                else:
                    stop_price = entry_price + ATR_MULT * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals