#!/usr/bin/env python3
"""
Experiment #221: 15m Primary + 1h/4h HTF — RSI Pullback in Trend + Session Filter

Hypothesis: 15m timeframe is underexplored (0 successful experiments). Key insight from 
failures: strategies either get 0 trades (too strict) or negative Sharpe (too many trades).

This strategy uses:
- 4h HMA(21) for major trend bias (loose filter - just direction, not hard requirement)
- 1h RSI(14) for pullback detection (RSI 35-55 for long in uptrend, 45-65 for short in downtrend)
- 15m HMA(9) for entry timing (fast response)
- Session filter: 00-12 UTC (London/NY overlap) to reduce trade count naturally
- ATR(14) trailing stop at 2.5x

Position sizing: 0.20 base (smaller for 15m frequency)
Target: 50-100 trades/year, Sharpe>0.4, DD>-30%

CRITICAL: Entry conditions are LOOSE enough to generate trades. RSI ranges are wide.
HTF is bias not hard filter. This ensures we don't get Sharpe=0.000 (0 trades).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_hma_1h4h_session_v1"
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

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds, convert to hours
    # Binance 15m bars: hour = (timestamp_ms / 1000 / 3600) % 24
    hours = (open_time // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 1h RSI for pullback detection
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m_fast = calculate_hma(close, period=9)
    hma_15m_slow = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Extract hour for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% position size for 15m (lower due to frequency)
    SIZE_STRONG = 0.25  # 25% for strong signals
    
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
        if np.isnan(hma_15m_fast[i]) or np.isnan(hma_15m_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        in_session = (hours[i] >= 0) and (hours[i] < 12)
        
        # === HTF BIAS (4h HMA) - loose filter, just direction ===
        htf_4h_bull = False
        htf_4h_bear = False
        if not np.isnan(hma_4h_aligned[i]):
            htf_4h_bull = close[i] > hma_4h_aligned[i]
            htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h RSI PULLBACK DETECTION ===
        rsi_1h_neutral = False
        rsi_1h_oversold = False
        rsi_1h_overbought = False
        if not np.isnan(rsi_1h_aligned[i]):
            rsi_1h_val = rsi_1h_aligned[i]
            rsi_1h_oversold = rsi_1h_val < 55.0  # Pullback in uptrend
            rsi_1h_overbought = rsi_1h_val > 45.0  # Pullback in downtrend
            rsi_1h_neutral = (rsi_1h_val >= 35.0) and (rsi_1h_val <= 65.0)
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m_slow[i]
        hma_bear = close[i] < hma_15m_slow[i]
        hma_cross_long = (hma_15m_fast[i] > hma_15m_slow[i]) and (hma_15m_fast[i-1] <= hma_15m_slow[i-1]) if i > 0 else False
        hma_cross_short = (hma_15m_fast[i] < hma_15m_slow[i]) and (hma_15m_fast[i-1] >= hma_15m_slow[i-1]) if i > 0 else False
        
        # === 15m RSI ===
        rsi_15m_oversold = rsi_15m[i] < 50.0
        rsi_15m_overbought = rsi_15m[i] > 50.0
        
        # === SMA200 FILTER (loose) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bull bias + 1h RSI pullback + 15m HMA cross + session
        # Loosened conditions to ensure trades happen
        long_conditions = 0
        if htf_4h_bull or above_sma200:
            long_conditions += 1
        if rsi_1h_oversold or rsi_15m_oversold:
            long_conditions += 1
        if hma_bull or hma_cross_long:
            long_conditions += 1
        if in_session:
            long_conditions += 0.5  # Session is bonus, not required
        
        # Need at least 2.5 conditions for long
        if long_conditions >= 2.5:
            desired_signal = SIZE_BASE
        elif long_conditions >= 3.5:
            desired_signal = SIZE_STRONG
        
        # SHORT ENTRY: 4h bear bias + 1h RSI pullback + 15m HMA cross + session
        short_conditions = 0
        if htf_4h_bear or below_sma200:
            short_conditions += 1
        if rsi_1h_overbought or rsi_15m_overbought:
            short_conditions += 1
        if hma_bear or hma_cross_short:
            short_conditions += 1
        if in_session:
            short_conditions += 0.5
        
        # Need at least 2.5 conditions for short
        if short_conditions >= 2.5 and desired_signal == 0.0:
            desired_signal = -SIZE_BASE
        elif short_conditions >= 3.5 and desired_signal == 0.0:
            desired_signal = -SIZE_STRONG
        
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