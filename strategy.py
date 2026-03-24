#!/usr/bin/env python3
"""
Experiment #773: 5m Primary + 15m/4h HTF — Mean Reversion Within Trend

Hypothesis: 5m is too noisy for pure trend following. Best approach is mean-reversion
entries WITHIN the HTF trend direction. Use 4h HMA for primary bias, 15m RSI for
momentum confirmation, 5m RSI pullback for precise entry timing.

Key innovations:
1. 4h HMA(21) for primary trend bias — only trade in HTF direction
2. 15m RSI(14) >50 for long bias, <50 for short bias — momentum confirmation
3. 5m RSI(7) pullback entries: long when RSI<45 in uptrend, short when RSI>55 in downtrend
4. Session filter: 08-20 UTC only (London/NY overlap = best liquidity)
5. Smaller size (0.15) due to higher trade frequency = less fee drag
6. Tighter stops (2x ATR) for faster turnover on 5m

Target: 50-120 trades/year, Sharpe>0.40, DD>-30%
Timeframe: 5m
Size: 0.15 discrete (0.0, ±0.15)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_rsi_pullback_15m4h_session_v1"
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

def get_session_mask(prices):
    """Create mask for trading hours 08-20 UTC only"""
    # Convert open_time to datetime and extract hour
    open_times = pd.to_datetime(prices['open_time'], unit='ms')
    hours = open_times.dt.hour
    # Session: 08-20 UTC (inclusive of 08, exclusive of 20)
    session_mask = (hours >= 8) & (hours < 20)
    return session_mask.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Session filter
    session_mask = get_session_mask(prices)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Smaller size for 5m due to higher trade frequency
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        
        # === SESSION FILTER ===
        if not session_mask[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI MOMENTUM CONFIRMATION ===
        rsi_15m_bull = rsi_15m_aligned[i] > 50.0
        rsi_15m_bear = rsi_15m_aligned[i] < 50.0
        
        # === 5m RSI PULLBACK ENTRIES ===
        # Long: HTF bull + 15m RSI bull + 5m RSI pullback oversold
        rsi_5m_oversold = rsi_7[i] < 45.0
        rsi_5m_overbought = rsi_7[i] > 55.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m RSI >50 + 5m RSI pullback <45
        if htf_4h_bull and rsi_15m_bull and rsi_5m_oversold:
            desired_signal = SIZE
        
        # SHORT: 4h bear + 15m RSI <50 + 5m RSI pullback >55
        elif htf_4h_bear and rsi_15m_bear and rsi_5m_overbought:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals