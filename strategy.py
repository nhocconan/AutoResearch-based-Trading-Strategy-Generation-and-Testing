#!/usr/bin/env python3
"""
Experiment #013: 4h Donchian Breakout + 12h Trend + Volume Filter

HYPOTHESIS: 4h Donchian(20) breakouts capture institutional momentum moves.
12h HMA(21) provides trend bias to avoid counter-trend trades. Volume spike
(>1.3x 20-bar MA) confirms genuine breakout vs fakeout. ATR(14) stoploss at
2.5x protects capital. This combination should work in bull (long breakouts
above 12h HMA) and bear (short breakdowns below 12h HMA).

TIMEFRAME: 4h primary
HTF: 12h for trend bias
TARGET: 75-200 total trades over 4 years (19-50/year)

KEY FIX FROM PREVIOUS FAILURES:
- Volume threshold lowered from 1.5x to 1.3x (too restrictive before)
- Added "price already outside channel" entry (not just breakout cross)
- Position persists across bars until explicit exit (not signal=0 every bar)
- Exit on opposite Donchian break OR 10-bar timeout OR stoploss
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_12h_trend_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20-period
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Previous Donchian for breakout detection
    donch_upper_prev = np.roll(donch_upper, 1)
    donch_lower_prev = np.roll(donch_lower, 1)
    donch_upper_prev[0] = np.nan
    donch_lower_prev[0] = np.nan
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout up: close crosses above previous upper band
        breakout_up = False
        if i > 1 and not np.isnan(donch_upper_prev[i]):
            breakout_up = (close[i] > donch_upper_prev[i]) and (close[i-1] <= donch_upper_prev[i-1])
        
        # Breakout down: close crosses below previous lower band
        breakout_down = False
        if i > 1 and not np.isnan(donch_lower_prev[i]):
            breakout_down = (close[i] < donch_lower_prev[i]) and (close[i-1] >= donch_lower_prev[i-1])
        
        # Price already outside channel (continuation entry)
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Breakout up OR price above upper channel + volume + bullish 12h trend
            if (breakout_up or price_above_upper) and vol_spike and price_above_12h_hma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Breakout down OR price below lower channel + volume + bearish 12h trend
            if (breakout_down or price_below_lower) and vol_spike and not price_above_12h_hma:
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
        
        # === EXIT: Opposite breakout ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: opposite breakout down
            if breakout_down:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: opposite breakout up
            if breakout_up:
                exit_triggered = True
        
        # === EXIT: Time-based (10 bars max hold) ===
        if in_position and (i - entry_bar) > 10:
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # If same direction, maintain position (don't churn)
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