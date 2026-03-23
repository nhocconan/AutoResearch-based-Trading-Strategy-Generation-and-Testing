#!/usr/bin/env python3
"""
Experiment #1365: 1h Primary + 4h/1d HTF — Simplified Multi-Path Entry

Hypothesis: Previous 1h strategies (#1355, #1358, #1360) failed with 0 trades due to 
OVER-FILTERING (too many confluence requirements). The 4h strategy #1354 had trades 
but negative Sharpe. Solution: Use 1h for entry timing within 4h/1d trend, but with 
MULTIPLE ENTRY PATHS to ensure trades happen. Key insight: fewer filters, more paths.

Key design choices:
1. 4h HMA(21) for primary trend — proven in #1354
2. 1d HMA(21) for macro bias — soft filter only (not required)
3. 1h RSI(14) with MODERATE bands (35-65) — not extreme, ensures trades
4. 1h ATR(14) trailing stop 2.0x — tighter than 4h for faster exits
5. Session filter (8-20 UTC) ONLY for new entries, not exits
6. Volume > 0.7x avg — lenient filter
7. Position size 0.25 — conservative for 1h volatility
8. MULTIPLE entry paths (3+) to ensure >=30 trades/train

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_atr_multipath_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

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
    
    # Calculate and align HTF HMA for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Extract hour from open_time for session filter
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        is_session = 8 <= hour_utc <= 20
        
        # Volume filter (lenient)
        vol_ok = not np.isnan(vol_sma[i]) and volume[i] > 0.7 * vol_sma[i]
        
        # === MACRO TREND (1d HMA) - soft filter only ===
        macro_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        macro_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === RSI MOMENTUM (MODERATE bands) ===
        rsi_bull = rsi[i] > 35.0
        rsi_bear = rsi[i] < 65.0
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        
        # === DESIRED SIGNAL — MULTIPLE PATHS ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (any one can trigger)
        # Path 1: Trend bull + RSI bull + session (primary)
        if trend_bull and rsi_bull and is_session:
            desired_signal = BASE_SIZE
        # Path 2: Trend bull + RSI oversold (pullback entry)
        elif trend_bull and rsi_oversold:
            desired_signal = BASE_SIZE
        # Path 3: Both HTF bull + volume ok (macro confirmation)
        elif trend_bull and macro_bull and vol_ok:
            desired_signal = BASE_SIZE * 0.7
        # Path 4: Simple trend follow (fallback to ensure trades)
        elif trend_bull and rsi[i] > 40.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS (any one can trigger)
        # Path 1: Trend bear + RSI bear + session (primary)
        elif trend_bear and rsi_bear and is_session:
            desired_signal = -BASE_SIZE
        # Path 2: Trend bear + RSI overbought (pullback entry)
        elif trend_bear and rsi_overbought:
            desired_signal = -BASE_SIZE
        # Path 3: Both HTF bear + volume ok (macro confirmation)
        elif trend_bear and macro_bear and vol_ok:
            desired_signal = -BASE_SIZE * 0.7
        # Path 4: Simple trend follow (fallback to ensure trades)
        elif trend_bear and rsi[i] < 60.0:
            desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.12:
            final_signal = BASE_SIZE
        elif desired_signal < -0.12:
            final_signal = -BASE_SIZE
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