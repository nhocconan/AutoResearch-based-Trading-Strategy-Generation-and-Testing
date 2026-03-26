#!/usr/bin/env python3
"""
Experiment #013: 12h Donchian Breakout + Volume + 1d Choppiness Filter

HYPOTHESIS: 12h Donchian(20) breakouts mark institutional accumulation/distribution.
Volume spike confirms the move is real. 1d Choppiness Index(14) filters out ranging
conditions where breakouts fail. In 2022 bear, Choppiness > 61.8 = range = no trades.
In bull markets and trending bears, Choppiness < 50 = enter breakouts.

KEY CHANGES from failed attempts:
- Choppiness filter eliminates 2022 range-bound whipsaws (main killer of Sharpe)
- Simple: only 3 conditions (breakout + volume + choppiness)
- Tighter volume threshold (1.5x) to reduce false breakouts

TIMEFRAME: 12h primary | HTF: 1d
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (Chop) - measures market trendiness
    CHOP > 61.8 = ranging (mean reversion environment)
    CHOP < 38.2 = trending (trend following environment)
    Values 38.2-61.8 = transitional
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of true range over period
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            atr_sum += tr
        
        # Highest - lowest over period
        hl_range = np.max(high[i - period + 1:i + 1]) - np.min(low[i - period + 1:i + 1])
        
        if hl_range > 1e-10 and atr_sum > 1e-10:
            # 100 * log10(atr_sum / hl_range) / log10(period)
            chop[i] = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d Choppiness for regime
    chop_1d_raw = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Previous Donchian for breakout detection
    donch_upper_prev = np.roll(donch_upper, 1)
    donch_lower_prev = np.roll(donch_lower, 1)
    donch_upper_prev[0] = np.nan
    donch_lower_prev[0] = np.nan
    
    # Volume MA (20 periods on 12h = ~10 days)
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
    
    warmup = 60  # Need enough for Donchian(20) + ATR(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER: 1d Choppiness ===
        # < 50 = trending (good for breakouts)
        # > 61.8 = ranging (skip - breakouts fail)
        chop_val = chop_1d_aligned[i]
        is_trending = chop_val < 50.0  # Only trade when trending
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_bullish = price_above_1d_hma if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout up: close crosses above previous upper band
        breakout_up = (close[i] > donch_upper_prev[i]) and \
                      (close[i-1] <= donch_upper_prev[i-1] if i > 1 else True)
        # Breakout down: close crosses below previous lower band
        breakout_down = (close[i] < donch_lower_prev[i]) and \
                         (close[i-1] >= donch_lower_prev[i-1] if i > 1 else True)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Only when trending, bullish HMA, volume spike, breakout up
            if is_trending and hma_bullish and vol_spike and breakout_up:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Only when trending, bearish HMA, volume spike, breakout down
            if is_trending and not hma_bullish and vol_spike and breakout_down:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Channel cross or opposite signal ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below lower channel
            if close[i] < donch_lower[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price breaks above upper channel
            if close[i] > donch_upper[i]:
                exit_triggered = True
        
        if exit_triggered:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals