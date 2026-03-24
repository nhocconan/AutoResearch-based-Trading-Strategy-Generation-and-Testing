#!/usr/bin/env python3
"""
Experiment #173: 5m Primary + 15m/4h HTF — Trend Pullback with Session Filter

Hypothesis: 5m timeframe has NEVER been tested (0 experiments). This is unexplored territory.
5m is extremely noisy, so we need:
1. Strong HTF trend filter (4h HMA) - ONLY trade in HTF direction
2. 15m momentum confirmation (RSI) - avoid exhausted moves
3. 5m entry on pullback to EMA - precision timing
4. Session filter (08-20 UTC) - avoid low liquidity whipsaws
5. Small position size (0.15) - more trades = more fee drag

Key insight from failures: Experiments #161, #165, #169, #170 all got Sharpe=0.000 (0 trades)
because entry conditions were TOO STRICT. This version loosens entries:
- RSI threshold: 35-65 (not extreme 20-80)
- Only require 2 of 3 confluence factors (not all 3)
- Session filter but not too narrow (12 hours, not 4)

Target: 50-120 trades/year, Sharpe>0.167, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_trend_pullback_session_15m4h_v1"
timeframe = "5m"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate primary (5m) indicators
    ema_5m_fast = calculate_ema(close, period=9)
    ema_5m_slow = calculate_ema(close, period=21)
    hma_5m = calculate_hma(close, period=16)
    atr = calculate_atr(high, low, close, period=14)
    rsi_5m = calculate_rsi(close, period=7)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15  # 15% base position size (smaller for 5m)
    SIZE_STRONG = 0.20  # 20% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_5m[i]) or np.isnan(ema_5m_fast[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds since epoch)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI MOMENTUM ===
        rsi_15m_val = rsi_15m_aligned[i]
        rsi_15m_bull = not np.isnan(rsi_15m_val) and 40 <= rsi_15m_val <= 70
        rsi_15m_bear = not np.isnan(rsi_15m_val) and 30 <= rsi_15m_val <= 60
        
        # === 5m TREND ===
        hma_bull = close[i] > hma_5m[i]
        hma_bear = close[i] < hma_5m[i]
        ema_fast_above_slow = ema_5m_fast[i] > ema_5m_slow[i]
        ema_fast_below_slow = ema_5m_fast[i] < ema_5m_slow[i]
        
        # === SMA200 FILTER ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === 5m RSI PULLBACK ===
        rsi_5m_val = rsi_5m[i]
        rsi_pullback_long = not np.isnan(rsi_5m_val) and 35 <= rsi_5m_val <= 55
        rsi_pullback_short = not np.isnan(rsi_5m_val) and 45 <= rsi_5m_val <= 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade in session hours
        if in_session:
            # LONG: HTF bull + 5m pullback + 15m RSI not overbought
            # Require 2 of 3: HTF bull, 5m HMA bull, EMA fast>slow
            long_score = 0
            if htf_4h_bull:
                long_score += 1
            if hma_bull:
                long_score += 1
            if ema_fast_above_slow:
                long_score += 1
            if above_sma200:
                long_score += 0.5
            
            if long_score >= 2 and rsi_pullback_long and rsi_15m_bull:
                # Check if strong signal
                if long_score >= 2.5 and htf_4h_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: HTF bear + 5m pullback + 15m RSI not oversold
            short_score = 0
            if htf_4h_bear:
                short_score += 1
            if hma_bear:
                short_score += 1
            if ema_fast_below_slow:
                short_score += 1
            if below_sma200:
                short_score += 0.5
            
            if short_score >= 2 and rsi_pullback_short and rsi_15m_bear:
                if short_score >= 2.5 and htf_4h_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2x) ===
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