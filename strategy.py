#!/usr/bin/env python3
"""
Experiment #1384: 4h Primary + 12h/1d HTF — Simplified Multi-Path Trend Following

Hypothesis: #1374 achieved Sharpe=0.224 using 4h Donchian + HMA + 12h/1d HTF + RSI + ATR.
Key insight: Clean trend following with multiple entry paths works better than regime filters.
This version ensures sufficient trade frequency (≥30 train, ≥3 test) by using 4 entry paths
per direction with relaxed filters and wider RSI bands.

Design:
1. 1d HMA(21) = macro trend bias
2. 12h HMA(21) = intermediate trend confirmation
3. 4h HMA(21) + slope = primary trend
4. Dual Donchian (20/50) breakout = entry triggers
5. RSI(14) 30-70 bands = momentum without over-filtering
6. ATR(14) trailing stop 2.5x = risk management
7. Position size 0.25-0.30 based on trend confluence
8. FOUR entry paths per direction = ensures >=30 trades/train

Target: 30-50 trades/year, Sharpe > 0.618, trades >= 30 train, >= 3 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_12h1d_rsi_atr_multipath_v2"
timeframe = "4h"
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

def calculate_hma_slope(hma, lookback=5):
    """HMA slope - positive = uptrend, negative = downtrend"""
    n = len(hma)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100.0
    return slope

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend filters
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_slope = calculate_hma_slope(hma_4h, lookback=5)
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_50_upper, donchian_50_lower = calculate_donchian(high, low, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    MAX_SIZE = 0.30
    
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
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(donchian_50_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_4h[i]) or np.isnan(hma_4h_slope[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h HMA) ===
        inter_bull = close[i] > hma_12h_aligned[i]
        inter_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA + slope) ===
        trend_bull = close[i] > hma_4h[i] and hma_4h_slope[i] > 0.0
        trend_bear = close[i] < hma_4h[i] and hma_4h_slope[i] < 0.0
        
        # === RSI MOMENTUM (wide bands for trade frequency) ===
        rsi_bull = rsi[i] > 30.0
        rsi_bear = rsi[i] < 70.0
        rsi_strong_bull = rsi[i] > 45.0
        rsi_strong_bear = rsi[i] < 55.0
        
        # === DUAL DONCHIAN BREAKOUT ===
        breakout_20_long = close[i] > donchian_20_upper[i-1]
        breakout_20_short = close[i] < donchian_20_lower[i-1]
        breakout_50_long = close[i] > donchian_50_upper[i-1]
        breakout_50_short = close[i] < donchian_50_lower[i-1]
        
        # === COUNT TREND CONFLUENCE (for position sizing) ===
        trend_confluence_long = sum([macro_bull, inter_bull, trend_bull])
        trend_confluence_short = sum([macro_bear, inter_bear, trend_bear])
        
        # === DESIRED SIGNAL - FOUR ENTRY PATHS PER DIRECTION ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (any one triggers entry)
        # Path 1: Donchian-20 breakout + trend + RSI (quick entry)
        if breakout_20_long and trend_bull and rsi_bull:
            size = MAX_SIZE if trend_confluence_long >= 3 else BASE_SIZE
            desired_signal = size
        # Path 2: Donchian-50 breakout + macro + RSI (strong breakout)
        elif breakout_50_long and macro_bull and rsi_strong_bull:
            size = MAX_SIZE if trend_confluence_long >= 2 else BASE_SIZE
            desired_signal = size
        # Path 3: Price above all HMAs + positive slope (trend continuation)
        elif close[i] > hma_4h[i] and close[i] > hma_12h_aligned[i] and hma_4h_slope[i] > 0.05:
            desired_signal = BASE_SIZE
        # Path 4: RSI momentum + above 4h HMA (momentum play - ensures trade frequency)
        elif rsi[i] > 50.0 and close[i] > hma_4h[i]:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRY PATHS (any one triggers entry)
        # Path 1: Donchian-20 breakout + trend + RSI (quick entry)
        elif breakout_20_short and trend_bear and rsi_bear:
            size = MAX_SIZE if trend_confluence_short >= 3 else BASE_SIZE
            desired_signal = -size
        # Path 2: Donchian-50 breakout + macro + RSI (strong breakout)
        elif breakout_50_short and macro_bear and rsi_strong_bear:
            size = MAX_SIZE if trend_confluence_short >= 2 else BASE_SIZE
            desired_signal = -size
        # Path 3: Price below all HMAs + negative slope (trend continuation)
        elif close[i] < hma_4h[i] and close[i] < hma_12h_aligned[i] and hma_4h_slope[i] < -0.05:
            desired_signal = -BASE_SIZE
        # Path 4: RSI weakness + below 4h HMA (momentum play - ensures trade frequency)
        elif rsi[i] < 50.0 and close[i] < hma_4h[i]:
            desired_signal = -BASE_SIZE
        
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
        if abs(desired_signal) >= BASE_SIZE * 0.8:
            if desired_signal > 0:
                final_signal = MAX_SIZE if desired_signal >= MAX_SIZE * 0.9 else BASE_SIZE
            else:
                final_signal = -MAX_SIZE if desired_signal <= -MAX_SIZE * 0.9 else -BASE_SIZE
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