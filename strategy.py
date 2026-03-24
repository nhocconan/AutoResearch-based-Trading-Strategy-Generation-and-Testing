#!/usr/bin/env python3
"""
Experiment #401: 15m Primary + 1h/4h/1d HTF — Simplified Multi-TF Trend Pullback

Hypothesis: Previous 15m/30m/1h strategies failed due to ZERO trades (Sharpe=0.000).
The entry conditions were too strict (5+ confluence filters that never all aligned).

This version SIMPLIFIES while keeping MTF edge:
1. 4h HMA = primary trend direction (only trade with 4h trend)
2. 1d HMA = higher-level bias filter (loose, just avoid counter-trend)
3. 15m RSI(7) = entry timing on pullbacks (RSI<45 in uptrend, RSI>55 in downtrend)
4. Session filter = UTC 00-12 only (London/NY overlap, higher quality moves)
5. ATR stoploss = 2.5x from entry

Key changes from failed experiments:
- LOOSENED RSI thresholds (45/55 instead of 30/70) for MORE trades
- Only 3 confluence requirements (not 5+)
- Removed Choppiness/ADX regime (too many false negatives)
- Session filter replaces complex regime detection

Target: 50-80 trades/year on 15m, Sharpe>0.5, DD<-30%
Position size: 0.20 base (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_simplified_hma_rsi_4h1d_session_v1"
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    hma_15m_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m entries
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
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
        
        # === SESSION FILTER (UTC 00-12 only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS (4h) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF BIAS (1d) - loose filter ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === HMA CROSSOVER (15m) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_15m_fast[i]) and not np.isnan(hma_15m_fast[i-1]):
            if not np.isnan(hma_15m[i]) and not np.isnan(hma_15m[i-1]):
                if hma_15m_fast[i-1] <= hma_15m[i-1] and hma_15m_fast[i] > hma_15m[i]:
                    hma_cross_long = True
                if hma_15m_fast[i-1] >= hma_15m[i-1] and hma_15m_fast[i] < hma_15m[i]:
                    hma_cross_short = True
        
        # === SMA200 FILTER (loose) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK (LOOSENED for more trades) ===
        # In uptrend: enter on RSI pullback to 40-50
        # In downtrend: enter on RSI bounce to 50-60
        rsi_pullback_long = rsi[i] < 50.0 and rsi[i] > 30.0
        rsi_pullback_short = rsi[i] > 50.0 and rsi[i] < 70.0
        
        # === ENTRY LOGIC (SIMPLIFIED - 3 confluence max) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m HMA bull + RSI pullback + session
        if htf_4h_bull and hma_bull and rsi_pullback_long:
            if in_session:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            elif hma_cross_long:
                # Allow breakout even outside session
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m HMA bear + RSI pullback + session
        elif htf_4h_bear and hma_bear and rsi_pullback_short:
            if in_session:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
            elif hma_cross_short:
                # Allow breakout even outside session
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