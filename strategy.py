#!/usr/bin/env python3
"""
Experiment #329: 15m Primary + 1h/1d HTF — Simplified HMA/RSI Momentum v1

Hypothesis: Previous 15m strategies failed with Sharpe=0.000 (ZERO trades) because
entry conditions were TOO STRICT. This version SIMPLIFIES logic to ensure trades.

Key changes from failed 15m experiments (#317, #319, #325):
1. REMOVED complex regime detection (Choppiness, CRSI) — too many filters = 0 trades
2. SIMPLIFIED to: 1d trend + 1h momentum + 15m HMA crossover
3. LOOSENED RSI thresholds: 40-60 instead of 30-70 (more signals)
4. Session filter is SOFT preference, not hard block
5. Smaller position size (0.15-0.25) for higher frequency tolerance

Entry Logic (SIMPLIFIED for trade generation):
- Long: 1d HMA50 bull + 1h RSI > 45 + 15m HMA10 crosses above HMA21
- Short: 1d HMA50 bear + 1h RSI < 55 + 15m HMA10 crosses below HMA21
- Session: prefer 00-12 UTC but allow all hours (soft filter)

Position sizing: 0.15 base, 0.25 when 1d+1h both aligned (discrete levels)
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.40, DD>-40%, trades>=50/year train, trades>=5/year test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_momentum_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m_fast = calculate_hma(close, period=10)
    hma_15m_slow = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_fast[i]) or np.isnan(hma_15m_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d TREND BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h MOMENTUM FILTER (LOOSENED for more trades) ===
        rsi_1h = rsi_1h_aligned[i]
        momentum_bull = rsi_1h > 45.0  # Was 50, loosened
        momentum_bear = rsi_1h < 55.0  # Was 50, loosened
        
        # === 15m HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0:
            # Fast crosses above slow
            if hma_15m_fast[i-1] <= hma_15m_slow[i-1] and hma_15m_fast[i] > hma_15m_slow[i]:
                hma_cross_long = True
            # Fast crosses below slow
            if hma_15m_fast[i-1] >= hma_15m_slow[i-1] and hma_15m_fast[i] < hma_15m_slow[i]:
                hma_cross_short = True
        
        # === 15m HMA POSITION ===
        hma_bull = close[i] > hma_15m_slow[i]
        hma_bear = close[i] < hma_15m_slow[i]
        
        # === SESSION FILTER (SOFT - prefer 00-12 UTC but allow all) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        preferred_session = (hour_utc >= 0 and hour_utc < 12)  # London+NY overlap
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        desired_signal = 0.0
        
        # Long entry: 1d bull + 1h momentum + 15m HMA cross or position
        if htf_1d_bull and momentum_bull:
            if hma_cross_long:
                # Crossover entry
                desired_signal = SIZE_STRONG if preferred_session else SIZE_BASE
            elif hma_bull and not in_position:
                # HMA position entry (catch trends without crossover)
                desired_signal = SIZE_BASE
        
        # Short entry: 1d bear + 1h momentum + 15m HMA cross or position
        elif htf_1d_bear and momentum_bear:
            if hma_cross_short:
                # Crossover entry
                desired_signal = -SIZE_STRONG if preferred_session else -SIZE_BASE
            elif hma_bear and not in_position:
                # HMA position entry
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals