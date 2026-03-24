#!/usr/bin/env python3
"""
Experiment #813: 5m Primary + 15m/4h HTF — Intraday Trend Following with Session Filter

Hypothesis: First-ever 5m experiment. 5m entries with 15m/4h HTF trend filter + 
London/NY session timing (08-20 UTC) will capture intraday momentum while avoiding 
overnight noise and Asian session chop. Key innovations:
1. 4h HMA(21) for primary HTF trend bias (stronger than 15m)
2. 15m HMA(16/48) for secondary trend confirmation
3. 5m RSI(14) pullback entries in trend direction (40-60 range, NOT extremes)
4. Session filter: only trade 08:00-20:00 UTC (London/NY overlap)
5. Small size 0.15 (5m = more trades = more fee drag)
6. Loose RSI thresholds to ensure ≥50 trades/year

Entry logic (LOOSE for trade generation):
- LONG: 4h HMA bull + 15m HMA bull + RSI(5m) 35-55 + session active
- SHORT: 4h HMA bear + 15m HMA bear + RSI(5m) 45-65 + session active

Target: Sharpe>0.40, trades>=50/year, DD>-30%
Timeframe: 5m
Size: 0.15 (smaller due to higher trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_hma_rsi_session_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if timestamp is within London/NY session (08:00-20:00 UTC)"""
    # open_time is in milliseconds since epoch
    dt = pd.to_datetime(open_time, unit='ms', utc=True)
    hour = dt.hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMAs
    hma_15m_16_raw = calculate_hma(df_15m['close'].values, period=16)
    hma_15m_48_raw = calculate_hma(df_15m['close'].values, period=48)
    hma_4h_21_raw = calculate_hma(df_4h['close'].values, period=21)
    
    hma_15m_16_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_16_raw)
    hma_15m_48_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_48_raw)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21_raw)
    
    # Calculate 5m indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Smaller size for 5m (more trades = more fees)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_16_aligned[i]) or np.isnan(hma_15m_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_21_aligned[i]
        htf_4h_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 15m HMA CROSSOVER ===
        hma_15m_bull = hma_15m_16_aligned[i] > hma_15m_48_aligned[i]
        hma_15m_bear = hma_15m_16_aligned[i] < hma_15m_48_aligned[i]
        
        # === 5m RSI PULLBACK (LOOSE thresholds for trade generation) ===
        # For longs: RSI pulled back to 35-55 range in uptrend
        # For shorts: RSI pulled back to 45-65 range in downtrend
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m bull + RSI pullback + session active
        if htf_4h_bull and hma_15m_bull and rsi_long_pullback and session_active:
            desired_signal = SIZE
        
        # SHORT: 4h bear + 15m bear + RSI pullback + session active
        elif htf_4h_bear and hma_15m_bear and rsi_short_pullback and session_active:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals