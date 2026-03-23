#!/usr/bin/env python3
"""
Experiment #1415: 1h Primary + 4h/1d HTF — Regime-Adaptive Pullback Strategy

Hypothesis: 1h timeframe with strict HTF filtering (4h + 1d HMA alignment) will
generate FEW trades (30-80/year) while capturing trend continuations. The key
insight from failed 1h strategies (#1405, #1408, #1410) is that lower TF needs
MUCH stricter HTF filters to avoid fee drag.

Design:
1. 1d HMA(21) = macro trend direction (ONLY trade with this)
2. 4h HMA(21) = intermediate confirmation (must align with 1d)
3. 1h RSI(14) pullback = entry timing (RSI 35-45 for long, 55-65 for short)
4. Session filter = only 8-20 UTC (reduces trades by ~50%)
5. Volume filter = >0.8x 20-bar average (confirms participation)
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.22 = conservative for 1h volatility

Why this differs from failed 1h strategies:
- #1405 used Choppiness + CRSI = too many signals, negative Sharpe
- #1408 had 0 trades = filters too strict
- This uses SIMPLE RSI pullback within STRONG HTF trend = fewer but higher quality trades

Target: 40-70 trades/year, Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 3 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_htf_trend_rsi_pullback_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - for pullback entry timing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.nanmean(volume[i-period+1:i+1])
    
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    # Convert to seconds, then to datetime
    timestamps = open_time / 1000.0
    # Extract hour from timestamp (UTC)
    hours = ((timestamps % 86400) / 3600).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (>0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === MACRO TREND (1d HMA) - strongest filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - must align with macro ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF TREND ALIGNMENT (BOTH must agree) ===
        htf_aligned_bull = macro_bull and inter_bull
        htf_aligned_bear = macro_bear and inter_bear
        
        # === RSI PULLBACK ENTRY (within HTF trend) ===
        # Long: RSI pulled back to 35-45 zone in bull trend
        rsi_pullback_long = 35.0 <= rsi[i] <= 48.0
        # Short: RSI pulled back to 55-65 zone in bear trend
        rsi_pullback_short = 52.0 <= rsi[i] <= 65.0
        
        # === RSI MOMENTUM CONFIRMATION ===
        # For long: RSI should be rising (current > previous)
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 and not np.isnan(rsi[i-1]) else False
        # For short: RSI should be falling
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 and not np.isnan(rsi[i-1]) else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + RSI pullback + session + volume
        if htf_aligned_bull and rsi_pullback_long and in_session and volume_ok:
            # Add RSI momentum confirmation for stronger signal
            if rsi_rising:
                desired_signal = BASE_SIZE
            else:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY: HTF bear + RSI pullback + session + volume
        elif htf_aligned_bear and rsi_pullback_short and in_session and volume_ok:
            # Add RSI momentum confirmation for stronger signal
            if rsi_falling:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals