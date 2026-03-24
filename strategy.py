#!/usr/bin/env python3
"""
Experiment #069: 15m Primary + 1h/1d HTF — Camarilla Pivot Mean Reversion + Session Filter

Hypothesis: After 68 failed experiments, 15m strategies keep failing due to:
- Too strict filters = 0 trades (experiments #057, #059, #061, #065)
- Too loose filters = fee drag from >300 trades/year

SOLUTION: Camarilla Pivot levels from 1d HTF + 1h HMA trend filter + RSI(7) timing
- Camarilla S3/R3 are proven mean-reversion zones in crypto (Pivot Boss methodology)
- 1h HMA provides intermediate trend bias (faster than 1d, slower than 15m noise)
- RSI(7) on 15m for entry timing (faster response than RSI(14))
- Session filter: 00-12 UTC (London/NY overlap = highest liquidity)
- Position size: 0.18 (smaller for 15m frequency)
- Target: 50-100 trades/year, Sharpe > 0.167 (beat current best)

Key design:
- LONG: price < Camarilla S3 + RSI(7) < 30 + 1h HMA bullish OR neutral + session 00-12 UTC
- SHORT: price > Camarilla R3 + RSI(7) > 70 + 1h HMA bearish OR neutral + session 00-12 UTC
- Stoploss: 2.0x ATR(14) trailing
- Discrete signals: 0.0, ±0.18 (minimize churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_session_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close, period=1):
    """
    Camarilla Pivot Levels
    Uses previous day's H, L, C to calculate intraday support/resistance
    R3/R4 = resistance, S3/S4 = support
    R4/S4 = breakout levels, R3/S3 = mean reversion levels
    """
    n = len(close)
    
    # Get previous day's high, low, close (simplified: use rolling period)
    prev_high = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    prev_close = pd.Series(close).shift(period).values
    
    r4 = np.zeros(n)
    r3 = np.zeros(n)
    s3 = np.zeros(n)
    s4 = np.zeros(n)
    r4[:] = np.nan
    r3[:] = np.nan
    s3[:] = np.nan
    s4[:] = np.nan
    
    for i in range(period + 1, n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        H = prev_high[i]
        L = prev_low[i]
        C = prev_close[i]
        range_hl = H - L
        
        if range_hl > 1e-10:
            r4[i] = C + range_hl * 1.5000  # breakout level
            r3[i] = C + range_hl * 1.0833  # mean reversion short
            s3[i] = C - range_hl * 1.0833  # mean reversion long
            s4[i] = C - range_hl * 1.5000  # breakdown level
    
    return r3, s3, r4, s4

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for intermediate trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 1d Camarilla pivots and align
    camarilla_1d = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=1
    )
    r3_1d_raw, s3_1d_raw, r4_1d_raw, s4_1d_raw = camarilla_1d
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_raw)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_raw)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d_raw)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)  # faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (conservative for 15m frequency)
    
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
        if np.isnan(hma_1h_aligned[i]):
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
        
        # === SESSION FILTER (00-12 UTC) ===
        # Extract hour from open_time (milliseconds since epoch)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        in_session = 0 <= hour_utc <= 12  # London + NY overlap
        
        # === HTF TREND BIAS (1h HMA) ===
        htf_bull = close[i] > hma_1h_aligned[i]
        htf_bear = close[i] < hma_1h_aligned[i]
        htf_neutral = abs(close[i] - hma_1h_aligned[i]) / (hma_1h_aligned[i] + 1e-10) < 0.01
        
        # === CAMARILLA LEVELS ===
        near_s3 = close[i] <= s3_aligned[i] * 1.002  # at or below S3
        near_r3 = close[i] >= r3_aligned[i] * 0.998  # at or above R3
        near_s4 = close[i] <= s4_aligned[i] * 1.002  # breakdown (avoid long)
        near_r4 = close[i] >= r4_aligned[i] * 0.998  # breakout (avoid short)
        
        # === RSI FILTER (15m timing) ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        rsi_extreme_oversold = rsi_7[i] < 20.0
        rsi_extreme_overbought = rsi_7[i] > 80.0
        
        # === DESIRED SIGNAL (Camarilla Mean Reversion) ===
        desired_signal = 0.0
        
        # LONG: price at S3 + RSI oversold + HTF not strongly bear + in session
        # Allow long if HTF is bull OR neutral (not strongly bear)
        if near_s3 and not near_s4 and in_session:
            if rsi_oversold and (htf_bull or htf_neutral):
                desired_signal = SIZE
            elif rsi_extreme_oversold:  # extreme oversold overrides HTF
                desired_signal = SIZE * 0.7
        
        # SHORT: price at R3 + RSI overbought + HTF not strongly bull + in session
        # Allow short if HTF is bear OR neutral (not strongly bull)
        if near_r3 and not near_r4 and in_session:
            if rsi_overbought and (htf_bear or htf_neutral):
                desired_signal = -SIZE
            elif rsi_extreme_overbought:  # extreme overbought overrides HTF
                desired_signal = -SIZE * 0.7
        
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