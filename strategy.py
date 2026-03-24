#!/usr/bin/env python3
"""
Experiment #433: 5m Primary + 15m/4h HTF — Session Filter + Trend Following

Hypothesis: 5m has ZERO prior experiments. Key insight: 5m needs HTF trend filter
to avoid whipsaw, session filter to avoid low-volume noise, and SIMPLE entries
(not complex regime detection which caused 0 trades in #421-#432).

Approach:
- 15m HMA(21) for primary trend direction
- 4h HMA(21) for higher timeframe bias (only trade with 4h trend)
- 5m RSI(14) for entry timing (pullback entries in trend direction)
- Session filter: 08:00-20:00 UTC only (high volume, avoid Asia low-vol)
- ATR(14) stoploss at 2.0x from entry

Why this might work:
- Previous failures used complex regime detection = 0 trades
- This uses SIMPLE trend + pullback = more trades, cleaner signals
- Session filter reduces noise during low-volume hours
- 4h HTF filter prevents counter-trend trades (major source of losses)

Target: Sharpe>0.40, DD>-35%, trades>=50/year (10-12/day during session)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_trend_rsi_15m4h_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def is_session_active(open_time_unix_ms, start_hour=8, end_hour=20):
    """Check if timestamp is within trading session (UTC)"""
    # Convert ms timestamp to datetime
    ts_seconds = open_time_unix_ms / 1000.0
    hour = pd.to_datetime(ts_seconds, unit='s').hour
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
    
    # Calculate and align HTF HMA for trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    hma_5m_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
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
        
        if np.isnan(hma_5m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
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
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        in_session = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === HTF TREND BIAS (4h + 15m must align) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # Strong bias when both HTF agree
        htf_strong_bull = htf_4h_bull and htf_15m_bull
        htf_strong_bear = htf_4h_bear and htf_15m_bear
        
        # === 5m LOCAL TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === HMA CROSSOVER (5m) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_5m_fast[i]) and not np.isnan(hma_5m_fast[i-1]):
            if not np.isnan(hma_5m[i]) and not np.isnan(hma_5m[i-1]):
                if hma_5m_fast[i-1] <= hma_5m[i-1] and hma_5m_fast[i] > hma_5m[i]:
                    hma_cross_long = True
                if hma_5m_fast[i-1] >= hma_5m[i-1] and hma_5m_fast[i] < hma_5m[i]:
                    hma_cross_short = True
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSE: 35/65 for more trades) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral = 35.0 <= rsi[i] <= 65.0
        
        # === ENTRY LOGIC (SIMPLE: trend + pullback) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session:
            # LONG: 4h+15m bull + 5m pullback (RSI oversold or HMA cross)
            if htf_strong_bull:
                # Entry on pullback in uptrend
                if rsi_oversold and above_sma50:
                    desired_signal = SIZE_STRONG
                elif hma_cross_long and above_sma50:
                    desired_signal = SIZE_BASE
                elif hma_5m_bull and rsi[i] < 50.0 and above_sma50:
                    desired_signal = SIZE_BASE
            
            # SHORT: 4h+15m bear + 5m retracement (RSI overbought or HMA cross)
            elif htf_strong_bear:
                # Entry on retracement in downtrend
                if rsi_overbought and below_sma50:
                    desired_signal = -SIZE_STRONG
                elif hma_cross_short and below_sma50:
                    desired_signal = -SIZE_BASE
                elif hma_5m_bear and rsi[i] > 50.0 and below_sma50:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals