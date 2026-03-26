#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume Confirmation + 1d Trend Filter

HYPOTHESIS: Simple Donchian(20) breakout on 4h captures institutional moves.
Combined with 1d HMA(48) trend direction filter and strict volume spike (>2x),
this will generate 75-150 total trades over 4 years — the sweet spot.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout works in ALL markets because it captures momentum bursts
- Bull: Long breakouts above 1d HMA with trailing ATR stop
- Bear: Short breakouts below 1d HMA
- ATR-based exits prevent large losses in whipsaws

KEY DIFFERENCE FROM FAILED STRATEGIES:
- Previous #015 had 1550 trades → overtrading = failure
- Previous #013 had 154 trades but negative Sharpe
- This strategy uses STRICTER volume filter (2x vs 1.5x) and single entry condition
- Donchian breakouts are rare (1-2 per month per direction) = natural trade limiter
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_tight_vol_1d_hma_v1"
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
    """Donchian Channel - upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, mid, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA(48) for trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_mid, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume: strict filter (2x average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    # Trade cooldown to prevent overtrading
    bars_since_entry = 999
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            bars_since_entry += 1
            continue
        
        # Update bars since last entry
        if in_position:
            bars_since_entry += 1
        
        # === TREND FILTER (1d HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        trend_bearish = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # === VOLUME FILTER (strict 2x) ===
        vol_confirm = vol_ratio[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        # Long: close breaks above 20-bar high
        long_breakout = close[i] > donchian_upper[i]
        # Short: close breaks below 20-bar low
        short_breakout = close[i] < donchian_lower[i]
        
        # === STOPLOSS CHECK (trailing ATR) ===
        stoploss_triggered = False
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if stoploss_triggered:
            desired_signal = 0.0
        else:
            # Require minimum 10 bars since last entry (cooldown)
            min_bars_cooldown = 10
            
            # LONG: Breakout + trend aligned + volume + cooldown
            if long_breakout and trend_bullish and vol_confirm and bars_since_entry >= min_bars_cooldown:
                desired_signal = SIZE
            
            # SHORT: Breakout + trend aligned + volume + cooldown
            elif short_breakout and trend_bearish and vol_confirm and bars_since_entry >= min_bars_cooldown:
                desired_signal = -SIZE
            
            # Hold current position
            elif in_position:
                desired_signal = position_side * SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
                if position_side > 0:
                    stop_price = close[i] - 2.0 * entry_atr
                else:
                    stop_price = close[i] + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals