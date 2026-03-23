#!/usr/bin/env python3
"""
Experiment #1258: 30m Primary + 4h HTF — Simplified Trend + RSI Pullback

Hypothesis: Recent 30m/1h strategies (#1248, #1250, #1251, #1255) all have Sharpe=0.000 = ZERO TRADES.
Entry conditions too strict when combining CRSI + Choppiness + multiple HTF filters.

This strategy SIMPLIFIES to ensure trades happen:
1. 4h HMA(21) for macro trend direction (proven in best strategies)
2. 30m RSI(14) for entry timing (simpler than CRSI, more reliable)
3. Volume filter (VERY loose: >0.3x 20-bar avg)
4. Session filter (WIDE: 6-22 UTC for liquidity)
5. RSI thresholds WIDE (30/70 vs 25/75) to ensure entries
6. ATR(14) stoploss at 2.5x
7. Signal size: 0.25 (conservative for 30m)

Key changes from failed experiments:
- Remove Choppiness Index (was blocking signals)
- Remove CRSI complexity (use simple RSI)
- WIDER RSI thresholds (30/70 vs 20/80)
- VERY loose volume filter (0.3x vs 0.8x)
- WIDE session window (6-22 UTC vs 8-20)
- No ADX requirement (was blocking)

Target: Sharpe > 0.612, trades >= 40/year, DD < -30%
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_4h_hma_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
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
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 30m indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume MA (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (6-22 UTC - WIDE window) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 6 <= hour_utc <= 22
        
        # === TREND FILTER (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (VERY loose) ===
        volume_ok = volume[i] > 0.3 * vol_ma[i]
        
        # === RSI ENTRY SIGNALS (WIDE thresholds for more trades) ===
        rsi_oversold = rsi[i] < 40.0  # Wide band for more entries
        rsi_overbought = rsi[i] > 60.0  # Wide band for more entries
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long: Trend bull + RSI oversold + volume + session
        if trend_bull and rsi_oversold and volume_ok and in_session:
            desired_signal = BASE_SIZE
        
        # Short: Trend bear + RSI overbought + volume + session
        elif trend_bear and rsi_overbought and volume_ok and in_session:
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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