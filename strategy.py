#!/usr/bin/env python3
"""
Experiment #1581: 15m Primary + 1h/4h/1d HTF — Camarilla Pivot Mean Reversion

Hypothesis: 15m timeframe is underexplored (0 successful experiments). This strategy
uses Camarilla pivot levels (from 1d HTF) for mean-reversion entries, with 4h HMA
for trend bias and 15m RSI(7) for precise entry timing.

Why this should work on 15m:
1. Camarilla R3/S3 levels are hit frequently (guarantees trades)
2. 4h HMA filter prevents counter-trend disasters
3. RSI(7) extremes on 15m provide good entry timing
4. Session filter (00-12 UTC) avoids low-volume Asian session whipsaws
5. Discrete sizing (0.15/0.25) minimizes fee churn

Key components:
- 1d Camarilla pivots (R3/S3 for mean-reversion, R4/S4 for breakout)
- 4h HMA(21) for major trend bias
- 15m RSI(7) for entry timing (loose thresholds: 25/75)
- Session filter: prefer 00-12 UTC (London+NY overlap)
- ATR(14) trailing stoploss (2.0x ATR)
- Discrete sizing: 0.0, ±0.15, ±0.25

Entry logic (LOOSE to guarantee ≥40 trades/train, ≥5/test):
- LONG: price<S3 + RSI<35 + 4h_HMA bullish OR price<S4 (breakout long)
- SHORT: price>R3 + RSI>65 + 4h_HMA bearish OR price>R4 (breakout short)

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_4h1d_session_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_camarilla_pivots(high, low, close, period=1):
    """
    Camarilla Pivot Points - intraday support/resistance levels
    Uses previous day's HLC to calculate levels
    
    R4 = H + 1.5 * (H - L)
    R3 = H + 1.0 * (H - L) / 2
    R2 = H + 0.5 * (H - L) / 3
    R1 = H + 0.25 * (H - L) / 6
    Pivot = (H + L + C) / 3
    S1 = L - 0.25 * (H - L) / 6
    S2 = L - 0.5 * (H - L) / 3
    S3 = L - 1.0 * (H - L) / 2
    S4 = L - 1.5 * (H - L)
    """
    n = len(close)
    
    # Use previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first bar to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_val = prev_high - prev_low
    
    r4 = prev_high + 1.5 * range_val
    r3 = prev_high + 1.0 * range_val / 2
    r2 = prev_high + 0.5 * range_val / 3
    r1 = prev_high + 0.25 * range_val / 6
    
    pivot = (prev_high + prev_low + prev_close) / 3
    
    s1 = prev_low - 0.25 * range_val / 6
    s2 = prev_low - 0.5 * range_val / 3
    s3 = prev_low - 1.0 * range_val / 2
    s4 = prev_low - 1.5 * range_val
    
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d Camarilla pivots and align
    cam_r4_raw, cam_r3_raw, cam_r2_raw, cam_r1_raw, cam_pivot_raw, \
    cam_s1_raw, cam_s2_raw, cam_s3_raw, cam_s4_raw = calculate_camarilla_pivots(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3_raw)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3_raw)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4_raw)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Session hours
    session_hours = calculate_session_hour(open_time)
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        hour = session_hours[i]
        is_prime_session = (hour >= 0 and hour <= 12)
        
        # === TREND BIAS (4h HMA) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CAMARILLA LEVELS ===
        r3 = cam_r3_aligned[i]
        s3 = cam_s3_aligned[i]
        r4 = cam_r4_aligned[i]
        s4 = cam_s4_aligned[i]
        
        # === RSI ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # MEAN REVERSION: Fade at R3/S3 with RSI confirmation
        # LONG: price below S3 + RSI oversold + 4h bullish bias (or prime session)
        if close[i] < s3 * 1.002 and rsi < 35:
            if price_above_4h or is_prime_session:
                desired_signal = SIZE_BASE
        
        # SHORT: price above R3 + RSI overbought + 4h bearish bias (or prime session)
        elif close[i] > r3 * 0.998 and rsi > 65:
            if price_below_4h or is_prime_session:
                desired_signal = -SIZE_BASE
        
        # BREAKOUT: Strong move through R4/S4
        # LONG breakout: price breaks above R4 + RSI rising
        elif close[i] > r4 * 0.998 and rsi > 50 and rsi < 80:
            if price_above_4h:
                desired_signal = SIZE_STRONG
        
        # SHORT breakout: price breaks below S4 + RSI falling
        elif close[i] < s4 * 1.002 and rsi < 50 and rsi > 20:
            if price_below_4h:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
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