#!/usr/bin/env python3
"""
Experiment #949: 15m Primary + 1h/1d HTF — HMA Trend + RSI Momentum + Session Filter

Hypothesis: 15m timeframe is underexplored (0 experiments so far). Key insight from failures:
- Previous 15m strategies had Sharpe=0.000 = ZERO TRADES (entry conditions too strict)
- Need LOOSE entry conditions to guarantee ≥10 trades/train, ≥3/test
- Use HTF (1d/1h) for trend bias, 15m for entry timing precision
- Session filter (UTC 00-12) preferred but NOT required to ensure trades

Key innovations:
1. 1d HMA(21) for primary trend bias (simple, proven effective)
2. 1h RSI(14) for momentum regime (long when >45, short when <55 - LOOSE)
3. 15m HMA(8/21) crossover for entry trigger (loose - just need alignment)
4. Session preference: UTC 00-12 (London/NY overlap) but not mandatory
5. ATR(14) 2.5x trailing stop for risk management
6. Small position sizes: 0.15-0.25 (15m has higher frequency)

Why this should work:
- LOOSE RSI thresholds (45/55 instead of 30/70) = more trades
- 1d trend filter prevents counter-trend trades in strong moves
- 15m HMA crossover gives precise entry within HTF trend
- Session filter is soft preference, not hard requirement

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    hma_15m_8 = calculate_hma(close, period=8)
    hma_15m_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time], dtype=np.int32)
    
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
        
        if np.isnan(hma_15m_8[i]) or np.isnan(hma_15m_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA trend) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h MOMENTUM (RSI - LOOSE THRESHOLDS) ===
        rsi_1h = rsi_1h_aligned[i]
        htf_1h_momentum_bull = rsi_1h > 45.0  # LOOSE: not requiring oversold
        htf_1h_momentum_bear = rsi_1h < 55.0  # LOOSE: not requiring overbought
        
        # === 15m HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_15m_8[i-1]) and not np.isnan(hma_15m_21[i-1]):
            hma_crossover_long = (hma_15m_8[i-1] <= hma_15m_21[i-1]) and (hma_15m_8[i] > hma_15m_21[i])
            hma_crossover_short = (hma_15m_8[i-1] >= hma_15m_21[i-1]) and (hma_15m_8[i] < hma_15m_21[i])
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_15m_8[i] > hma_15m_21[i]
        hma_15m_bear = hma_15m_8[i] < hma_15m_21[i]
        
        # === SESSION FILTER (PREFERRED BUT NOT REQUIRED) ===
        utc_hour = utc_hours[i]
        session_preferred = (utc_hour >= 0 and utc_hour <= 12)  # London/NY overlap
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries (HTF bullish bias + momentum + 15m trigger)
        if htf_1d_bull and htf_1h_momentum_bull:
            # Crossover entry (stronger signal)
            if hma_crossover_long:
                if session_preferred:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Trend continuation entry (looser)
            elif hma_15m_bull:
                desired_signal = SIZE_BASE
        
        # SHORT entries (HTF bearish bias + momentum + 15m trigger)
        elif htf_1d_bear and htf_1h_momentum_bear:
            # Crossover entry (stronger signal)
            if hma_crossover_short:
                if session_preferred:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Trend continuation entry (looser)
            elif hma_15m_bear:
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