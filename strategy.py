#!/usr/bin/env python3
"""
Experiment #733: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe with strict session filtering (08-20 UTC) and multi-TF
confluence can capture intraday momentum while avoiding noise during low-volume
hours. 4h HMA provides primary trend bias, 15m RSI confirms momentum, 5m entries
on pullbacks in trend direction.

Key innovations:
1. 4h HMA(21) for primary trend bias — only trade in HTF trend direction
2. 15m RSI(14) for intermediate momentum confirmation
3. 5m RSI(7) for precise pullback entries (faster than 14)
4. Session filter: only 08:00-20:00 UTC (high volume, avoid Asia night)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.25 (smaller due to more trades)
7. Volume confirmation: taker_buy_volume > SMA(20) for entry validation

Entry conditions:
- LONG: 4h HMA bull + 15m RSI > 50 + 5m RSI < 40 (pullback) + session + volume
- SHORT: 4h HMA bear + 15m RSI < 50 + 5m RSI > 60 (pullback) + session + volume

Target: Sharpe>0.40, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15-0.25 discrete (smaller due to higher trade frequency)
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return sma

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if timestamp is within active trading session (UTC)"""
    # open_time is in milliseconds since epoch
    timestamp = pd.to_datetime(open_time, unit='ms', utc=True)
    hour = timestamp.hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
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
    
    # Calculate 5m indicators
    rsi_5m_fast = calculate_rsi(close, period=7)
    rsi_5m_slow = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(rsi_5m_fast[i]) or np.isnan(rsi_5m_slow[i]):
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
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = taker_buy_volume[i] > vol_sma_20[i] * 0.5
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM (intermediate confirmation) ===
        rsi_15m_bull = rsi_15m_aligned[i] > 50.0
        rsi_15m_bear = rsi_15m_aligned[i] < 50.0
        rsi_15m_strong_bull = rsi_15m_aligned[i] > 55.0
        rsi_15m_strong_bear = rsi_15m_aligned[i] < 45.0
        
        # === 5m PULLBACK ENTRY (fast RSI for timing) ===
        rsi_5m_oversold = rsi_5m_fast[i] < 35.0
        rsi_5m_overbought = rsi_5m_fast[i] > 65.0
        rsi_5m_extreme_oversold = rsi_5m_fast[i] < 25.0
        rsi_5m_extreme_overbought = rsi_5m_fast[i] > 75.0
        
        # === ENTRY LOGIC (strict confluence for 5m) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m momentum + 5m pullback + session + volume
        if htf_4h_bull and rsi_15m_bull and session_active:
            if rsi_5m_oversold and volume_above_avg:
                if rsi_5m_extreme_oversold or rsi_15m_strong_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Also enter on 5m RSI crossing back above 40 from oversold
            elif i > 0 and not np.isnan(rsi_5m_fast[i-1]):
                if rsi_5m_fast[i-1] < 35.0 and rsi_5m_fast[i] > 35.0 and volume_above_avg:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m momentum + 5m pullback + session + volume
        elif htf_4h_bear and rsi_15m_bear and session_active:
            if rsi_5m_overbought and volume_above_avg:
                if rsi_5m_extreme_overbought or rsi_15m_strong_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Also enter on 5m RSI crossing back below 60 from overbought
            elif i > 0 and not np.isnan(rsi_5m_fast[i-1]):
                if rsi_5m_fast[i-1] > 65.0 and rsi_5m_fast[i] < 65.0 and volume_above_avg:
                    desired_signal = -SIZE_BASE
        
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