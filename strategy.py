#!/usr/bin/env python3
"""
Experiment #029: 15m Primary + 1h/1d HTF — Camarilla Pivot + RSI Mean Reversion + Session Filter

Hypothesis: 15m strategies fail due to either too many trades (fee drag) or too few (0 Sharpe).
Solution: Use 1d Camarilla pivot levels for key S/R zones, 1h HMA for trend bias, 15m RSI for entry timing.
- Camarilla R3/S3 = mean reversion zones (fade extremes)
- Camarilla R4/S4 = breakout zones (follow with trend)
- Session filter: Only trade 00-12 UTC (London+NY overlap, highest crypto volume)
- 3+ confluence required: HTF trend + pivot level + RSI extreme + session
- Position size: 0.18 (conservative for 15m frequency)
- Target: 50-80 trades/year, Sharpe > 0.2

Key design choices:
- Timeframe: 15m (use HTF for direction, 15m for precise entry)
- HTF: 1d Camarilla pivots + 1h HMA trend
- Entry: RSI(7) extremes at pivot levels with session filter
- Stoploss: 2.0x ATR trailing (tighter for 15m)
- LOOSE enough filters to ensure >=30 trades on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_session_1h1d_v1"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_camarilla_pivots(open_prev, high_prev, low_prev, close_prev):
    """
    Camarilla Pivot Points
    R4/S4 = breakout levels, R3/S3 = mean reversion levels
    Formula based on previous day's OHLC
    """
    range_hl = high_prev - low_prev
    
    r4 = close_prev + range_hl * 1.5000
    r3 = close_prev + range_hl * 1.2500
    r2 = close_prev + range_hl * 1.1666
    r1 = close_prev + range_hl * 1.0833
    
    s4 = close_prev - range_hl * 1.5000
    s3 = close_prev - range_hl * 1.2500
    s2 = close_prev - range_hl * 1.1666
    s1 = close_prev - range_hl * 1.0833
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d Camarilla pivots and align
    n_1d = len(df_1d)
    r4_1d = np.zeros(n_1d)
    r3_1d = np.zeros(n_1d)
    s3_1d = np.zeros(n_1d)
    s4_1d = np.zeros(n_1d)
    
    for i in range(1, n_1d):
        r4_1d[i], r3_1d[i], _, _, _, _, _, s3_1d[i], s4_1d[i] = calculate_camarilla_pivots(
            df_1d['open'].values[i-1],
            df_1d['high'].values[i-1],
            df_1d['low'].values[i-1],
            df_1d['close'].values[i-1]
        )
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (conservative for 15m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(hma_15m[i]) or np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour <= 12)
        
        # === HTF TREND BIAS ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === CAMARILLA PIVOT LEVELS ===
        # Near S3 = oversold zone (mean reversion long)
        # Near R3 = overbought zone (mean reversion short)
        # Break above R4 = bullish breakout
        # Break below S4 = bearish breakout
        near_s3 = close[i] <= s3_aligned[i] * 1.002  # within 0.2% of S3
        near_r3 = close[i] >= r3_aligned[i] * 0.998  # within 0.2% of R3
        breakout_r4 = close[i] > r4_aligned[i]
        breakout_s4 = close[i] < s4_aligned[i]
        
        # === RSI EXTREMES (LOOSE to ensure trades) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # === DESIRED SIGNAL (Multiple Entry Patterns) ===
        desired_signal = 0.0
        
        # Pattern 1: Mean reversion at S3 + RSI oversold + HTF not bearish
        if near_s3 and rsi_oversold and not htf_1d_bear and in_session:
            desired_signal = SIZE
        
        # Pattern 2: Mean reversion at R3 + RSI overbought + HTF not bullish
        elif near_r3 and rsi_overbought and not htf_1d_bull and in_session:
            desired_signal = -SIZE
        
        # Pattern 3: Breakout above R4 + HTF bull + RSI confirming
        elif breakout_r4 and htf_1h_bull and rsi_7[i] > 50.0 and in_session:
            desired_signal = SIZE * 0.7
        
        # Pattern 4: Breakout below S4 + HTF bear + RSI confirming
        elif breakout_s4 and htf_1h_bear and rsi_7[i] < 50.0 and in_session:
            desired_signal = -SIZE * 0.7
        
        # Pattern 5: Extreme RSI mean reversion (any session)
        elif rsi_extreme_oversold and hma_15m_bull and not htf_1d_bear:
            desired_signal = SIZE * 0.5
        
        elif rsi_extreme_overbought and hma_15m_bear and not htf_1d_bull:
            desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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