#!/usr/bin/env python3
"""
Experiment #453: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe is unexplored (0 experiments). Key insights:
1. 5m has extreme noise - MUST use HTF (4h/15m) for trend direction
2. Session filter CRITICAL: only trade 07-21 UTC (London/NY = best liquidity)
3. Size must be smaller (0.15-0.18) due to higher trade frequency = fee drag
4. Entry: 4h HMA trend + 15m RSI momentum + 5m EMA pullback (3-confluence)
5. Exit: ATR trailing stop (1.8x) + time-based exit (max 48 hours)

Why this might work:
- 4h HMA provides strong trend bias (proven in #435)
- 15m RSI filters entries (avoid chasing exhausted moves)
- 5m EMA pullback = precise entry timing on retracements
- Session filter avoids Asian session whipsaw (00-06 UTC)
- Smaller size (0.18) controls fee impact from frequent trades

Target: Sharpe>0.5, DD>-30%, trades=60-150/year, all symbols positive Sharpe
Timeframe: 5m (NEW - first 5m experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_htf_trend_ema_pullback_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate 15m RSI for momentum
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate primary (5m) indicators
    ema_5m_fast = calculate_ema(close, period=9)
    ema_5m_slow = calculate_ema(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.18  # Smaller size for 5m (more trades = fee drag)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    entry_bar = 0
    max_hold_bars = 576  # 48 hours at 5m = 576 bars
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_15m_aligned[i]) or np.isnan(ema_5m_fast[i]) or np.isnan(ema_5m_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (07-21 UTC only) ===
        # Convert open_time (milliseconds) to hour UTC
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        
        in_session = 7 <= hour_utc <= 21
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 15m must agree) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        htf_bull = htf_4h_bull and htf_15m_bull
        htf_bear = htf_4h_bear and htf_15m_bear
        
        # === 15m RSI MOMENTUM FILTER ===
        rsi_15m = rsi_15m_aligned[i]
        # Bullish: RSI between 45-70 (not overbought, but positive momentum)
        rsi_bull = 45.0 < rsi_15m < 70.0
        # Bearish: RSI between 30-55 (not oversold, but negative momentum)
        rsi_bear = 30.0 < rsi_15m < 55.0
        
        # === 5m EMA PULLBACK ENTRY ===
        ema_bull = ema_5m_fast[i] > ema_5m_slow[i]
        ema_bear = ema_5m_fast[i] < ema_5m_slow[i]
        
        # Pullback to slow EMA (within 0.3%)
        pullback_long = low[i] <= ema_5m_slow[i] * 1.003
        pullback_short = high[i] >= ema_5m_slow[i] * 0.997
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 1.0
        vol_confirmed = vol_ratio > 0.7  # At least 70% of average volume
        
        # === ENTRY LOGIC (3-CONFLUENCE) ===
        desired_signal = 0.0
        
        # Long: 4h bull + 15m bull + 15m RSI ok + 5m EMA bull + pullback + volume
        if htf_bull and rsi_bull and ema_bull and pullback_long and vol_confirmed:
            desired_signal = SIZE
        
        # Short: 4h bear + 15m bear + 15m RSI ok + 5m EMA bear + pullback + volume
        elif htf_bear and rsi_bear and ema_bear and pullback_short and vol_confirmed:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (1.8x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        # === TIME-BASED EXIT (max 48 hours) ===
        time_exit = False
        if in_position and (i - entry_bar) > max_hold_bars:
            time_exit = True
        
        if stoploss_triggered or time_exit:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL ===
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
                entry_atr = atr[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 1.8 * entry_atr
                else:
                    stop_price = entry_price + 1.8 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals