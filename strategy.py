#!/usr/bin/env python3
"""
Experiment #361: 15m Primary + 1h/4h/1d HTF — HMA Pullback with Session Filter v1

Hypothesis: Previous 15m strategies failed due to overly strict entry conditions
(0 trades = Sharpe=0.000). This version SIMPLIFIES entries while using proven
MTF framework: 4h HMA for trend direction, 15m HMA pullback for entry timing.

Key design choices:
1. 4h HMA(21) determines trend bias (long only when 4h HMA bull, short when bear)
2. 15m HMA(21) pullback entry (price touches HMA in trend direction)
3. RSI(7) extreme for timing (oversold in uptrend, overbought in downtrend)
4. Session filter: 00-12 UTC only (London+NY overlap, reduces trade count)
5. 1d HMA for additional regime confirmation
6. Position size: 0.15-0.20 (smaller for 15m frequency)
7. Stoploss: 2.0x ATR(14) from entry

Target: 50-100 trades/year, Sharpe>0.4, DD>-35%, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_pullback_session_4h1d_v1"
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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

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
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour < 12)
        
        # === HTF TREND BIAS (4h + 1d) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when 4h and 1d agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === HMA PULLBACK DETECTION ===
        # Price pulled back to HMA (within 0.5% of HMA)
        hma_pullback_long = False
        hma_pullback_short = False
        
        if not np.isnan(hma_15m[i]) and hma_15m[i] > 1e-10:
            pullback_threshold = 0.005  # 0.5%
            
            # Long: price was above HMA, now touching/near HMA from above
            if i > 0 and not np.isnan(hma_15m[i-1]):
                was_above = close[i-1] > hma_15m[i-1]
                now_near = abs(close[i] - hma_15m[i]) / hma_15m[i] < pullback_threshold
                hma_pullback_long = was_above and now_near and hma_bull
            
            # Short: price was below HMA, now touching/near HMA from below
            if i > 0 and not np.isnan(hma_15m[i-1]):
                was_below = close[i-1] < hma_15m[i-1]
                now_near = abs(close[i] - hma_15m[i]) / hma_15m[i] < pullback_threshold
                hma_pullback_short = was_below and now_near and hma_bear
        
        # === RSI EXTREMES (faster for 15m) ===
        rsi_oversold = rsi[i] < 35.0  # Loosened for more trades
        rsi_overbought = rsi[i] > 65.0  # Loosened for more trades
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === ENTRY LOGIC (SIMPLIFIED for trade generation) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + pullback to 15m HMA + RSI oversold + session
        if htf_strong_bull and in_session:
            if hma_pullback_long and rsi_oversold and above_sma200:
                desired_signal = SIZE_STRONG
            elif hma_bull and rsi_oversold and above_sma200:
                # Simpler entry: just HMA bull + RSI oversold
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + pullback to 15m HMA + RSI overbought + session
        elif htf_strong_bear and in_session:
            if hma_pullback_short and rsi_overbought and below_sma200:
                desired_signal = -SIZE_STRONG
            elif hma_bear and rsi_overbought and below_sma200:
                # Simpler entry: just HMA bear + RSI overbought
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