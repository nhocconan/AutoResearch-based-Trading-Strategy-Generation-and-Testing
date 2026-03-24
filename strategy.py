#!/usr/bin/env python3
"""
Experiment #293: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback v1

Hypothesis: 5m timeframe needs EXTREME selectivity. Use 4h/15m HMA for trend direction,
only take 5m RSI pullback entries in established trend direction. Session filter (08-20 UTC)
eliminates low-liquidity Asian session whipsaws. This gives HTF trade frequency with
5m entry precision.

Key design:
1. 4h HMA(21) = primary trend bias (long only above, short only below)
2. 15m HMA(21) = secondary confirmation (must align with 4h)
3. 5m RSI(7) pullback = entry timing (RSI<35 in uptrend, RSI>65 in downtrend)
4. Session filter = only 08-20 UTC (avoid Asian session chop)
5. ATR(14) stoploss = 2.5x from entry
6. Size = 0.15 (smaller due to higher trade frequency on 5m)

Why this might work:
- 5m alone = too many false signals (whipsaw city)
- 4h alone = too few trades, misses entry precision
- 4h trend + 5m entry = best of both worlds
- Session filter removes 40% of low-quality bars (Asian session)
- RSI pullback = buys dips in uptrend, sells rallies in downtrend

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_15m4h_v1"
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

def is_session_active(open_time, start_hour=8, end_hour=20):
    """
    Check if bar is within active session (08-20 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    ts_seconds = open_time / 1000.0
    hour_utc = (ts_seconds % 86400) / 3600.0
    
    return start_hour <= hour_utc < end_hour

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
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 5m entries
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Smaller size for 5m (more trades = more fee drag)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Track consecutive signals to avoid churn
    prev_signal = 0.0
    signal_streak = 0
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_15m_aligned[i]):
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
        
        # === SESSION FILTER (08-20 UTC) ===
        in_session = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === 4h TREND BIAS (primary) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m TREND CONFIRMATION (secondary) ===
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # === 5m HMA TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === SMA200 FILTER (long-term bias) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # In uptrend: wait for RSI pullback to oversold (but not extreme crash)
        rsi_pullback_long = rsi[i] < 35.0 and rsi[i] > 15.0
        # In downtrend: wait for RSI rally to overbought (but not extreme spike)
        rsi_pullback_short = rsi[i] > 65.0 and rsi[i] < 85.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m bull + RSI pullback + in session
        if htf_4h_bull and htf_15m_bull and rsi_pullback_long and in_session:
            # Extra confirmation: above SMA200 for major uptrend
            if above_sma200:
                desired_signal = SIZE
            else:
                # Weaker signal if below SMA200 (might be counter-trend rally)
                desired_signal = SIZE * 0.5
        
        # SHORT: 4h bear + 15m bear + RSI pullback + in session
        elif htf_4h_bear and htf_15m_bear and rsi_pullback_short and in_session:
            # Extra confirmation: below SMA200 for major downtrend
            if below_sma200:
                desired_signal = -SIZE
            else:
                # Weaker signal if above SMA200 (might be counter-trend dip)
                desired_signal = -SIZE * 0.5
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.4:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === SIGNAL CHURN FILTER ===
        # Only change signal if it persists for 2 consecutive bars
        # This reduces false entries on 5m noise
        if final_signal != 0.0 and final_signal == prev_signal:
            signal_streak += 1
        elif final_signal != 0.0 and final_signal != prev_signal:
            signal_streak = 1
        else:
            signal_streak = 0
        
        # Require 2-bar confirmation for new signals (but not for exits)
        if final_signal != 0.0 and signal_streak < 2 and prev_signal == 0.0:
            final_signal = 0.0  # Wait for confirmation
        
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
        prev_signal = final_signal
    
    return signals