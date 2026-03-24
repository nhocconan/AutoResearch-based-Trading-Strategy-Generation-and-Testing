#!/usr/bin/env python3
"""
Experiment #209: 15m Primary + 1h/1d HTF — Volume-Confirmed Pullback Strategy

Hypothesis: 15m timeframe has ZERO successful experiments because previous attempts
were too strict (0 trades). This version uses LOOSER entry conditions to ensure
trades actually trigger while maintaining quality through HTF filter + volume confirm.

Key Differences from Failed 15m Attempts:
- RSI(7) threshold: <35/>65 (not <25/>75 which never triggers)
- 1h HMA trend filter (more responsive than 4h/1d for 15m entries)
- Volume confirmation: current volume > 0.8 * 20-bar avg (not 1.5x which is too strict)
- Session filter: 00-12 UTC only (reduces noise but still allows trades)
- Position size: 0.20 (smaller for 15m frequency, reduces fee impact)

Entry Logic:
- LONG: 1h HMA bullish + 15m RSI(7)<35 + volume confirm + session window
- SHORT: 1h HMA bearish + 15m RSI(7)>65 + volume confirm + session window

Exit: RSI crosses midpoint (50) OR stoploss at 2.5x ATR

Target: Sharpe>0.399 (beat current best), trades>=40 train, trades>=5 test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_rsi_pullback_1h_session_v1"
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
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for trend bias (more responsive than 4h for 15m)
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d HMA for major trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma20 = calculate_sma(volume, 20)
    hma_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% position size for 15m frequency
    SIZE_STRONG = 0.25  # 25% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h_aligned[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma20[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)  # London + NY overlap
        
        # === HTF BIAS (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === HTF MAJOR TREND (1d HMA) ===
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 0.8 * vol_sma20[i]  # At least 80% of avg volume
        
        # === RSI VALUES (pullback detection) ===
        rsi_oversold = rsi_7[i] < 35.0  # Looser threshold for more trades
        rsi_overbought = rsi_7[i] > 65.0  # Looser threshold for more trades
        rsi_neutral = rsi_7[i] >= 35.0 and rsi_7[i] <= 65.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_21[i]
        hma_bear = close[i] < hma_21[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1h bullish + RSI oversold + volume confirm + session
        # Use OR logic for 1d filter to allow more trades
        if htf_1h_bull and rsi_oversold and vol_confirm:
            # Require either 1d bull OR in_session (not both - too strict)
            if htf_1d_bull or in_session:
                # Strong signal if both 1d bull AND in session
                if htf_1d_bull and in_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 1h bearish + RSI overbought + volume confirm + session
        elif htf_1h_bear and rsi_overbought and vol_confirm:
            # Require either 1d bear OR in_session (not both - too strict)
            if htf_1d_bear or in_session:
                # Strong signal if both 1d bear AND in session
                if htf_1d_bear and in_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC (RSI crosses midpoint) ===
        if in_position and position_side > 0:
            # Long exit: RSI crosses above 55 (momentum fading)
            if rsi_7[i] > 55.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Short exit: RSI crosses below 45 (momentum fading)
            if rsi_7[i] < 45.0:
                desired_signal = 0.0
        
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