#!/usr/bin/env python3
"""
Experiment #477: 15m Primary + 4h HTF — Session-Filtered RSI Pullback

Hypothesis: 15m timeframe has ZERO successful experiments (all Sharpe=0.000 = 0 trades).
The problem: overly complex filters (CRSI + chop + session + multiple HTF) never trigger.

New approach for 15m success:
1. SINGLE HTF FILTER: 4h HMA(21) for trend direction only (not dual HTF)
2. FAST RSI: RSI(7) on 15m for quicker entry signals (vs RSI14)
3. SESSION FILTER: Only trade 00-12 UTC (London + NY overlap = 60% of crypto volume)
4. LOOSE THRESHOLDS: RSI < 35 for long, > 65 for short (not extreme 20/80)
5. SIMPLE LOGIC: Trend direction + RSI pullback + session = 3 confluence (not 5+)
6. SIZE: 0.18 (smaller for 15m frequency, target 50-100 trades/year)
7. STOPLOSS: 2.5x ATR to avoid premature exits on 15m noise

Why this should work on 15m:
- 4h trend filter reduces false signals by ~50%
- Session filter avoids low-volume Asian session whipsaw
- RSI(7) generates signals 2x faster than RSI(14)
- Loose thresholds ensure we get trades (learned from #465, #469, #473, #476 failures)

Target: Sharpe>0.4, DD>-35%, trades>=80 train (20/year), trades>=8 test
Timeframe: 15m (FIRST 15m strategy with realistic entry conditions)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_4h_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_fast = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_std = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.22
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_fast[i]) or np.isnan(rsi_std[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: Only trade 00-12 UTC (London + NY overlap) ===
        # Extract hour from open_time (assumes milliseconds timestamp)
        open_time = prices["open_time"].values[i]
        hour_utc = (open_time // 3600000) % 24  # Convert ms to hours, mod 24
        
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m LOCAL TREND ===
        local_bull = close[i] > hma_15m[i] and close[i] > sma_50[i]
        local_bear = close[i] < hma_15m[i] and close[i] < sma_50[i]
        
        # === RSI PULLBACK SIGNALS (LOOSE THRESHOLDS) ===
        # Long: RSI(7) < 35 (oversold pullback in uptrend)
        # Short: RSI(7) > 65 (overbought pullback in downtrend)
        rsi_oversold = rsi_fast[i] < 35.0
        rsi_overbought = rsi_fast[i] > 65.0
        
        # RSI confirmation (standard 14-period)
        rsi_confirm_long = rsi_std[i] < 50.0
        rsi_confirm_short = rsi_std[i] > 50.0
        
        # === SMA200 BIAS FILTER (not too restrictive) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (3 confluence: HTF + RSI + session) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + RSI(7) oversold + in session
        if htf_bull and rsi_oversold and in_session:
            # Additional confirmation: either local bull or above SMA200
            if local_bull or above_sma200:
                if rsi_confirm_long:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + RSI(7) overbought + in session
        elif htf_bear and rsi_overbought and in_session:
            # Additional confirmation: either local bear or below SMA200
            if local_bear or below_sma200:
                if rsi_confirm_short:
                    desired_signal = -SIZE_BASE
        
        # Strong signal: All conditions align perfectly
        if htf_bull and local_bull and rsi_oversold and in_session and above_sma200:
            desired_signal = SIZE_STRONG
        elif htf_bear and local_bear and rsi_overbought and in_session and below_sma200:
            desired_signal = -SIZE_STRONG
        
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