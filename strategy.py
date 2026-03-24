#!/usr/bin/env python3
"""
Experiment #045: 1h Primary + 4h/1d HTF — Simplified RSI Pullback with Loose Thresholds

Hypothesis: Previous 1h/30m strategies failed with 0 trades due to OVERLY STRICT filters.
This strategy SIMPLIFIES entry conditions to guarantee trade generation:
1. RSI(14) with LOOSE thresholds (25/75 instead of 15/85) — catches more reversals
2. 4h HMA trend ONLY (not 4h+1d) — fewer conflicting signals
3. NO Choppiness Index filter — it was blocking too many valid entries
4. Volume filter is LENIENT (>0.5x avg, not >0.8x)
5. Session filter is OPTIONAL (boosts signal but not required)
6. Discrete sizing: 0.25 base, 0.30 with HTF alignment

Key changes from failed experiments:
- RSI thresholds 25/75 (was 15/85 or 30/70 with too many filters)
- Only require 4h trend alignment (was 4h+1d+1w = too restrictive)
- Remove CHOP regime filter entirely (was blocking 60%+ of signals)
- Add ATR volatility filter to avoid dead markets (ATR > 0.3% of price)

Entry Logic:
- Long: RSI(14) < 25 + price > 4h_HMA + ATR > 0.3%
- Short: RSI(14) > 75 + price < 4h_HMA + ATR > 0.3%
- Size: 0.25 base, 0.30 if 4h HMA slope confirms

Risk: 2.5x ATR trailing stop, max signal 0.35, discrete levels
Target: Sharpe>0.4, trades>40/symbol train, >5/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_loose_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=20):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h HMA slope (trend direction)
    hma_4h_slope = np.full(n, np.nan)
    for i in range(5, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-5]):
            if hma_4h_aligned[i-5] > 1e-10:
                hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-5]) / hma_4h_aligned[i-5]
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume SMA for volume filter
    vol_sma = calculate_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.30
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # ATR as % of price (volatility filter)
        atr_pct = atr[i] / close[i] if close[i] > 1e-10 else 0.0
        
        # Volume filter (lenient: >0.5x average)
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 1.0
        vol_ok = vol_ratio > 0.5
        
        # Session filter (8-20 UTC) — boosts signal but not required
        hour_utc = (open_time[i] // 3600000) % 24
        is_session = 8 <= hour_utc <= 20
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_4h_slope_bull = hma_4h_slope[i] > 0.0005 if not np.isnan(hma_4h_slope[i]) else False
        hma_4h_slope_bear = hma_4h_slope[i] < -0.0005 if not np.isnan(hma_4h_slope[i]) else False
        
        # === ENTRY SIGNALS (LOOSE thresholds for trade generation) ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG: RSI < 25 (oversold) + price > 4h HMA (uptrend pullback)
        # Relaxed: only need ONE of the HTF conditions
        if rsi[i] < 25.0 and atr_pct > 0.003 and vol_ok:
            if hma_4h_bull or hma_4h_slope_bull:
                if hma_4h_bull and hma_4h_slope_bull:
                    signal_strength = CONFIRMED_SIZE
                else:
                    signal_strength = BASE_SIZE
                desired_signal = signal_strength
                if is_session:
                    desired_signal = min(desired_signal + 0.03, MAX_SIZE)
        
        # SHORT: RSI > 75 (overbought) + price < 4h HMA (downtrend rally)
        elif rsi[i] > 75.0 and atr_pct > 0.003 and vol_ok:
            if hma_4h_bear or hma_4h_slope_bear:
                if hma_4h_bear and hma_4h_slope_bear:
                    signal_strength = CONFIRMED_SIZE
                else:
                    signal_strength = BASE_SIZE
                desired_signal = -signal_strength
                if is_session:
                    desired_signal = max(desired_signal - 0.03, -MAX_SIZE)
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= CONFIRMED_SIZE * 0.85:
            final_signal = CONFIRMED_SIZE
        elif desired_signal <= -CONFIRMED_SIZE * 0.85:
            final_signal = -CONFIRMED_SIZE
        elif desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * BASE_SIZE
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