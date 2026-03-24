#!/usr/bin/env python3
"""
Experiment #309: 15m Primary + 1h/1d HTF — Selective Pullback with Volume Confirmation v1

Hypothesis: 15m strategies fail due to too many trades (fee drag). Solution:
1. Use 1h HMA as STRONG trend filter (only trade in HTF direction)
2. Use 1d HMA for major trend bias (boost size when aligned)
3. 15m RSI(7) for pullback entries within HTF trend (mean-reversion entry, trend exit)
4. Volume spike confirmation on entry bar (>1.5x 20-bar avg volume)
5. Session filter: only 00-12 UTC (London+NY overlap, highest liquidity)
6. ATR(14) stoploss at 2.5x from entry

This combines HTF trend direction (1h/1d) with 15m pullback timing.
Expected trade frequency: 40-80 trades/year (strict 3+ confluence required).
Position size: 0.15 base, 0.25 when 1d aligned (smaller for 15m frequency).

Key difference from failed #301: Added VOLUME confirmation + stricter HTF filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pullback_volume_session_1h1d_v1"
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

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above rolling average"""
    n = len(volume)
    spike = np.zeros(n, dtype=bool)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if vol_avg[i] > 1e-10 and volume[i] > threshold * vol_avg[i]:
            spike[i] = True
    
    return spike

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    hours = np.zeros(len(prices), dtype=int)
    for i in range(len(prices)):
        ts = prices['open_time'].iloc[i]
        # Convert to hours since epoch, then mod 24
        hours[i] = (ts // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for pullback detection
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI
    sma_200 = calculate_sma(close, 200)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # Session hours (00-12 UTC = London+NY overlap)
    hours = get_hour_from_open_time(prices)
    in_session = (hours >= 0) & (hours < 12)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15  # Smaller for 15m frequency
    SIZE_STRONG = 0.25  # When 1d HTF aligned
    
    # Position tracking
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
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === MAJOR TREND BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI(7) oversold (<30) but price still above 1h HMA (trend intact)
        rsi_oversold = rsi_7[i] < 30.0
        # Short: RSI(7) overbought (>70) but price still below 1h HMA (trend intact)
        rsi_overbought = rsi_7[i] > 70.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume_spike[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 1h bull + 15m pullback (RSI<30) + volume spike + session
        # Confluence: htf_1h_bull + rsi_oversold + vol_confirmed + session_ok
        if htf_1h_bull and rsi_oversold and vol_confirmed and session_ok:
            # Bonus: if 1d also bull, use larger size
            if htf_1d_bull and above_sma200:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1h bear + 15m pullback (RSI>70) + volume spike + session
        # Confluence: htf_1h_bear + rsi_overbought + vol_confirmed + session_ok
        elif htf_1h_bear and rsi_overbought and vol_confirmed and session_ok:
            # Bonus: if 1d also bear, use larger size
            if htf_1d_bear and below_sma200:
                desired_signal = -SIZE_STRONG
            else:
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