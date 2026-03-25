#!/usr/bin/env python3
"""
Experiment #1529: 15m Primary + 1h/1d HTF — Daily Pivot + Session Filter + RSI Mean Reversion

Hypothesis: 15m timeframe can work with EXTREME selectivity. Key innovations:
1. Daily CPR (Central Pivot Range) from 1d HTF defines key support/resistance levels
2. 1h HMA(21) provides intraday trend bias (faster than 4h for 15m entries)
3. 15m RSI(7) for fast mean-reversion signals (oversold bounce in uptrend)
4. SESSION FILTER: Only trade 00-12 UTC (London+NY overlap = 70% of crypto volume)
5. Size: 0.15-0.20 (smaller for 15m frequency control)
6. Target: 50-80 trades/year (strict confluence = fewer but higher quality)

Why this should work on 15m:
- Session filter cuts trades by ~50% (no Asian session chop)
- 1h trend filter prevents counter-trend mean reversion disasters
- CPR levels are self-fulfilling (widely watched by algos)
- RSI(7) catches intraday extremes faster than RSI(14)
- Discrete sizing (0.0, ±0.15, ±0.20) minimizes fee churn

Entry confluence (ALL required):
- LONG: 1h_HMA bullish + price > Daily_Pivot + RSI(7) < 25 + UTC 00-12
- SHORT: 1h_HMA bearish + price < Daily_Pivot + RSI(7) > 75 + UTC 00-12

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%, trades/year<100
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_daily_pivot_1h_rsi7_session_v1"
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

def calculate_daily_pivot(open_price, high, low, close):
    """
    Daily Central Pivot Range (CPR)
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = (High + Low) / 2 + (High - Low) * 0.1
    Pivot = (High + Low + Close) / 3
    
    Returns: pivot, bc, tc arrays aligned to daily bars
    """
    n = len(close)
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            pivot[i] = (high[i] + low[i] + close[i]) / 3.0
            bc[i] = (high[i] + low[i]) / 2.0
            tc[i] = bc[i] + (high[i] - low[i]) * 0.1
    
    return pivot, bc, tc

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    utc_hour = (ts_seconds % 86400) / 3600.0
    return int(utc_hour)

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
    
    # Calculate and align 1h indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d CPR
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_daily_pivot(
        df_1d['open'].values,
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Session filter: UTC 00-12 (London + NY overlap)
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    is_session = (utc_hours >= 0) & (utc_hours <= 12)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(bc_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (CRITICAL for 15m) ===
        in_session = is_session[i]
        
        # === 1h TREND BIAS ===
        hma_1h_bullish = close[i] > hma_1h_aligned[i]
        hma_1h_bearish = close[i] < hma_1h_aligned[i]
        
        # === DAILY PIVOT POSITION ===
        above_pivot = close[i] > pivot_1d_aligned[i]
        below_pivot = close[i] < pivot_1d_aligned[i]
        
        # === CPR NARROW (consolidation before breakout) ===
        cpr_width = tc_1d_aligned[i] - bc_1d_aligned[i]
        avg_cpr_width = np.nanmean(tc_1d_aligned[max(0,i-20):i+1] - bc_1d_aligned[max(0,i-20):i+1])
        cpr_narrow = not np.isnan(avg_cpr_width) and cpr_width < avg_cpr_width * 0.7
        
        # === RSI EXTREMES (fast mean reversion) ===
        rsi_oversold = rsi_7[i] < 28
        rsi_overbought = rsi_7[i] > 72
        
        # === RSI DIVERGENCE CHECK (optional confluence) ===
        rsi_rising = rsi_7[i] > rsi_7[i-3] if i >= 3 and not np.isnan(rsi_7[i-3]) else False
        rsi_falling = rsi_7[i] < rsi_7[i-3] if i >= 3 and not np.isnan(rsi_7[i-3]) else False
        
        # === ENTRY LOGIC (ALL CONDITIONS REQUIRED - VERY SELECTIVE) ===
        desired_signal = 0.0
        
        # LONG: 1h bullish + above pivot + RSI oversold + in session + (CPR narrow OR RSI rising)
        if hma_1h_bullish and above_pivot and rsi_oversold and in_session:
            if cpr_narrow or rsi_rising:
                desired_signal = SIZE_BASE
                # Stronger signal if price near BC support
                if close[i] < bc_1d_aligned[i] * 1.005:
                    desired_signal = SIZE_STRONG
        
        # SHORT: 1h bearish + below pivot + RSI overbought + in session + (CPR narrow OR RSI falling)
        elif hma_1h_bearish and below_pivot and rsi_overbought and in_session:
            if cpr_narrow or rsi_falling:
                desired_signal = -SIZE_BASE
                # Stronger signal if price near TC resistance
                if close[i] > tc_1d_aligned[i] * 0.995:
                    desired_signal = -SIZE_STRONG
        
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