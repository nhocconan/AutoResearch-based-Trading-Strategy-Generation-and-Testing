#!/usr/bin/env python3
"""
Experiment #109: 15m Primary + 1h/1d HTF — Camarilla Pivot + RSI(7) + Choppiness + Session Filter

Hypothesis: After 108 failed experiments, 15m strategies fail due to:
(1) Too many trades → fee drag destroys PnL, OR (2) Too strict → 0 trades.

SOLUTION: Ultra-selective 15m entries with 3+ confluence filters:
- 1d Camarilla pivots (R3/S3 mean reversion, R4/S4 breakout)
- 1h HMA for intraday trend bias
- 15m RSI(7) for fast entry timing (oversold/overbought extremes)
- Choppiness Index to detect range vs trend regime
- Session filter: ONLY 00-12 UTC (London/NY overlap = 70% of crypto volume)

Key design choices:
- Timeframe: 15m (target 40-100 trades/year with strict filters)
- HTF: 1d Camarilla pivots + 1h HMA trend
- Entry: RSI(7)<20 or >80 + at pivot level + HTF bias + session
- Regime: CHOP>55 = mean revert at R3/S3, CHOP<55 = breakout at R4/S4
- Position size: 0.20 (20% of capital, smaller for 15m frequency)
- Stoploss: 2.0x ATR trailing (tighter for 15m swings)
- Session: only trade bars where hour UTC is 0-12

Target: Sharpe>0.167, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi7_chop_session_1h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Points
    R4/S4 = breakout levels, R3/S3 = mean reversion levels
    """
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    
    r4 = close + range_hl * 1.1 / 2.0
    r3 = close + range_hl * 1.1 / 4.0
    r2 = close + range_hl * 1.1 / 6.0
    r1 = close + range_hl * 1.1 / 12.0
    
    s4 = close - range_hl * 1.1 / 2.0
    s3 = close - range_hl * 1.1 / 4.0
    s2 = close - range_hl * 1.1 / 6.0
    s1 = close - range_hl * 1.1 / 12.0
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    seconds = open_time / 1000.0
    # Convert to UTC hour
    import datetime
    dt = datetime.datetime.utcfromtimestamp(seconds)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for intraday trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d Camarilla pivots and align
    # Use previous day's OHLC for pivot calculation
    d_close_1d = df_1d['close'].values
    d_high_1d = df_1d['high'].values
    d_low_1d = df_1d['low'].values
    d_open_1d = df_1d['open'].values
    
    # Shift by 1 to use completed day's data
    prev_d_close = np.roll(d_close_1d, 1)
    prev_d_close[0] = d_close_1d[0]
    prev_d_high = np.roll(d_high_1d, 1)
    prev_d_high[0] = d_high_1d[0]
    prev_d_low = np.roll(d_low_1d, 1)
    prev_d_low[0] = d_low_1d[0]
    
    r4_1d, r3_1d, r2_1d, r1_1d, pivot_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        prev_d_high, prev_d_low, prev_d_close, np.roll(prev_d_close, 1)
    )
    
    # Align 1d pivots to 15m
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # 15m HMA for local trend
    hma_15m = calculate_hma(close, period=13)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    
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
        if np.isnan(rsi_7[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
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
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour <= 12)
        
        if not in_session:
            # Close position if out of session
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # === HTF BIAS (1h and 1d HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias: both 1h and 1d agree
        strong_bull = htf_1h_bull and htf_1d_bull
        strong_bear = htf_1h_bear and htf_1d_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === CAMARILLA PIVOT LEVELS ===
        near_s3 = abs(close[i] - s3_aligned[i]) / (atr[i] + 1e-10) < 1.5
        near_r3 = abs(close[i] - r3_aligned[i]) / (atr[i] + 1e-10) < 1.5
        near_s4 = close[i] <= s4_aligned[i] + atr[i] * 0.5
        near_r4 = close[i] >= r4_aligned[i] - atr[i] * 0.5
        
        # === RSI(7) EXTREMES (faster than RSI(14)) ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        rsi_extreme_oversold = rsi_7[i] < 15.0
        rsi_extreme_overbought = rsi_7[i] > 85.0
        
        # === 15m HMA LOCAL TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === DESIRED SIGNAL (3+ Confluence Required) ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean revert at R3/S3
            # LONG: near S3 + RSI oversold + HTF not strongly bear + session
            if near_s3 and rsi_oversold and not strong_bear:
                desired_signal = SIZE
            # SHORT: near R3 + RSI overbought + HTF not strongly bull
            elif near_r3 and rsi_overbought and not strong_bull:
                desired_signal = -SIZE
            # Extreme mean reversion (override HTF)
            elif rsi_extreme_oversold and near_s3:
                desired_signal = SIZE * 0.7
            elif rsi_extreme_overbought and near_r3:
                desired_signal = -SIZE * 0.7
        else:
            # TREND REGIME: Breakout at R4/S4 with HTF bias
            # LONG: breakout above R4 + strong bull + RSI ok
            if near_r4 and strong_bull and rsi_7[i] > 30.0 and rsi_7[i] < 80.0:
                desired_signal = SIZE
            # SHORT: breakdown below S4 + strong bear + RSI ok
            elif near_s4 and strong_bear and rsi_7[i] < 70.0 and rsi_7[i] > 20.0:
                desired_signal = -SIZE
            # Pullback entry in trend
            elif strong_bull and rsi_oversold and hma_15m_bull:
                desired_signal = SIZE * 0.7
            elif strong_bear and rsi_overbought and hma_15m_bear:
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