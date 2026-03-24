#!/usr/bin/env python3
"""
Experiment #193: 5m Primary + 15m/4h HTF — Session Pullback Strategy

Hypothesis: 5m timeframe with strict session filter + HTF trend alignment can capture
intraday pullbacks within established trends. Previous lower TF strategies failed due
to either 0 trades (too strict) or excessive trades (fee drag). This version:

1. 4h HMA(21) for major trend bias — ONLY trade in HTF trend direction
2. 15m RSI(14) for intermediate momentum confirmation
3. 5m RSI(7) for pullback entry timing (oversold in uptrend, overbought in downtrend)
4. Session filter: 08-20 UTC (London/NY active hours) — avoids dead Asian session
5. Volatility filter: ATR ratio > 0.8 — avoid extremely low vol periods
6. Position size: 0.15-0.20 (smaller for 5m due to higher trade frequency)
7. Stoploss: 2.5x ATR trailing

Target: 60-100 trades/year, Sharpe>0.4, DD>-30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_pullback_rsi_15m4h_v1"
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

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for major trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 15m RSI for intermediate momentum
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate primary (5m) indicators
    rsi_5m = calculate_rsi(close, period=7)  # Faster RSI for 5m entries
    rsi_5m_slow = calculate_rsi(close, period=14)  # Standard RSI
    atr = calculate_atr(high, low, close, period=14)
    atr_long = calculate_atr(high, low, close, period=30)  # For vol filter
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_5m[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # London open (08) to NY close (20) — most liquid hours
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLATILITY FILTER ===
        # Avoid extremely low vol periods (ATR ratio < 0.8)
        vol_filter = True
        if not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            vol_ratio = atr[i] / atr_long[i]
            vol_filter = vol_ratio > 0.7
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM ===
        rsi_15m_neutral = True
        if not np.isnan(rsi_15m_aligned[i]):
            # RSI between 40-60 = neutral momentum (good for pullback entries)
            # RSI > 60 = bullish momentum
            # RSI < 40 = bearish momentum
            rsi_15m_neutral = (rsi_15m_aligned[i] >= 35) and (rsi_15m_aligned[i] <= 65)
            rsi_15m_bull = rsi_15m_aligned[i] > 50
            rsi_15m_bear = rsi_15m_aligned[i] < 50
        else:
            rsi_15m_bull = False
            rsi_15m_bear = False
        
        # === 5m PULLBACK SIGNALS ===
        rsi_5m_oversold = False
        rsi_5m_overbought = False
        if not np.isnan(rsi_5m[i]):
            rsi_5m_oversold = rsi_5m[i] < 35  # Pullback in uptrend
            rsi_5m_overbought = rsi_5m[i] > 65  # Pullback in downtrend
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m neutral/bull + 5m RSI oversold + session + vol
        if htf_4h_bull and in_session and vol_filter:
            if rsi_5m_oversold and above_sma50:
                # Strong signal if 15m also bullish
                if rsi_15m_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m neutral/bear + 5m RSI overbought + session + vol
        elif htf_4h_bear and in_session and vol_filter:
            if rsi_5m_overbought and below_sma50:
                # Strong signal if 15m also bearish
                if rsi_15m_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
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
        
        # === SESSION EXIT (close positions outside session) ===
        if in_position and not in_session:
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