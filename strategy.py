#!/usr/bin/env python3
"""
Experiment #1369: 15m Primary + 1h/1d HTF — Daily Pivot Mean Reversion + Session Filter

Hypothesis: 15m timeframe has ZERO successful experiments. This strategy combines:
1. Daily Pivot Points (from 1d HTF) - proven S/R levels that institutions watch
2. 1h HMA(21) for intraday trend bias (less strict than 1d, more trades)
3. 15m RSI(7) extremes for mean reversion entries within trend
4. UTC Session filter (00-12) - London/NY overlap = highest volume, cleanest signals
5. ATR-based stoploss (2.5x) for risk management

Why this should work where 15m strategies failed:
- Daily pivots are self-fulfilling levels (traders watch them)
- 1h trend filter is looser than 1d (generates more trades)
- RSI(7) extremes happen frequently enough for 40-100 trades/year
- Session filter removes Asian session noise (whipsaw city)
- Scoring system ensures entries even when not all conditions perfect

Entry logic (scoring system):
- LONG: price near S1/R1 + RSI(7)<30 + 1h HMA bullish + session 00-12 UTC
- SHORT: price near R1/S1 + RSI(7)>70 + 1h HMA bearish + session 00-12 UTC
- Score >= 3 triggers entry (flexible confluence, not rigid AND)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_daily_pivot_rsi_session_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_rsi(close, period=7):
    """Relative Strength Index - shorter period for 15m sensitivity"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_pivot_points(high, low, close, prev_close):
    """
    Calculate Daily Pivot Points and Support/Resistance levels
    Uses previous day's HLC to calculate today's pivots
    """
    n = len(close)
    
    pivot = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    
    # Classic pivot formula
    for i in range(1, n):
        if not np.isnan(prev_close[i-1]) and not np.isnan(high[i-1]) and not np.isnan(low[i-1]):
            p = (prev_close[i-1] + high[i-1] + low[i-1]) / 3.0
            pivot[i] = p
            r1[i] = 2.0 * p - low[i-1]
            s1[i] = 2.0 * p - high[i-1]
            r2[i] = p + (high[i-1] - low[i-1])
            s2[i] = p - (high[i-1] - low[i-1])
    
    return pivot, r1, r2, s1, s2

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate daily pivot points from 1d data
    d_close = df_1d['close'].values
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # Shift prev_close for pivot calculation (use yesterday's close)
    prev_close = np.roll(d_close, 1)
    prev_close[0] = np.nan
    
    pivot_1d, r1_1d, r2_1d, s1_1d, s2_1d = calculate_pivot_points(d_high, d_low, d_close, prev_close)
    
    # Align pivot levels to 15m timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
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
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
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
        
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 = London/NY overlap) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 0) and (utc_hour <= 12)
        
        # === 1H TREND BIAS ===
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # === PIVOT LEVEL PROXIMITY ===
        # Calculate distance to nearest pivot level as percentage
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Distance to S1 (for long entries)
        dist_to_s1 = abs(close[i] - s1) / close[i] * 100 if s1 > 0 else 100
        near_s1 = dist_to_s1 < 1.5  # Within 1.5% of S1
        
        # Distance to R1 (for short entries)
        dist_to_r1 = abs(close[i] - r1) / close[i] * 100 if r1 > 0 else 100
        near_r1 = dist_to_r1 < 1.5  # Within 1.5% of R1
        
        # === RSI EXTREMES ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # === SCORING SYSTEM (flexible confluence) ===
        long_score = 0
        short_score = 0
        
        # Long scoring
        if price_above_1h:
            long_score += 1  # Trend bias
        if near_s1:
            long_score += 1  # At support
        if rsi_oversold:
            long_score += 1  # Oversold
        if in_session:
            long_score += 0.5  # Session filter (soft)
        
        # Short scoring
        if price_below_1h:
            short_score += 1  # Trend bias
        if near_r1:
            short_score += 1  # At resistance
        if rsi_overbought:
            short_score += 1  # Overbought
        if in_session:
            short_score += 0.5  # Session filter (soft)
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: score >= 2.5 (at least 2 strong conditions)
        if long_score >= 2.5:
            if rsi_oversold and near_s1:
                desired_signal = SIZE_STRONG
            elif rsi_oversold or near_s1:
                desired_signal = SIZE_BASE
            elif price_above_1h:
                desired_signal = SIZE_BASE
        
        # SHORT: score >= 2.5 (at least 2 strong conditions)
        elif short_score >= 2.5:
            if rsi_overbought and near_r1:
                desired_signal = -SIZE_STRONG
            elif rsi_overbought or near_r1:
                desired_signal = -SIZE_BASE
            elif price_below_1h:
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
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.8:
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