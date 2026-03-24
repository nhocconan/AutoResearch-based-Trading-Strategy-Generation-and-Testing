#!/usr/bin/env python3
"""
Experiment #1029: 15m Primary + 1h/1d HTF — Daily CPR Breakout + Session Filter

Hypothesis: Daily Central Pivot Range (CPR) from 1d HTF provides strong S/R levels.
Combining CPR breakout with 1h HMA trend filter and 15m RSI momentum, restricted to
high-volume session (00-12 UTC), will generate selective high-quality trades.

Key innovations:
1. Daily CPR (Central Pivot Range): TC/BC/Pivot from previous 1d bar
   - TC = (H+L)/2, BC = (H+L+C)/3, Pivot = (TC+BC)/2
   - Narrow CPR (<1% of price) = potential breakout day
2. 1h HMA(21) for intermediate trend direction
3. 15m RSI(7) for entry momentum (not extreme, just confirmation)
4. Session filter: only 00-12 UTC (London+NY overlap = highest volume)
5. Very selective: need CPR breakout + 1h trend + RSI confirmation + session
6. Small size (0.15-0.20) for 15m frequency target (40-100 trades/year)

Why this should work:
- CPR levels are watched by institutional traders (self-fulfilling S/R)
- Session filter avoids low-volume Asian session whipsaws
- 1h trend filter prevents counter-trend CPR breaks
- RSI(7) confirms momentum without being too laggy
- Discrete sizing minimizes fee churn on signal changes

Entry conditions (balanced for trades):
- LONG: price>TC + narrow_CPR + 1h_HMA_bull + RSI(7)>50 + session
- SHORT: price<BC + narrow_CPR + 1h_HMA_bear + RSI(7)<50 + session

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_session_hma_rsi_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_cpr_from_htf(prices, df_htf):
    """
    Calculate Daily CPR (Central Pivot Range) from HTF data
    TC = (High + Low) / 2
    BC = (High + Low + Close) / 3
    Pivot = (TC + BC) / 2
    CPR Width = (TC - BC) / Pivot (normalized)
    
    Returns aligned arrays for 15m bars
    """
    n_ltf = len(prices)
    
    # Extract HTF OHLC
    htf_high = df_htf['high'].values
    htf_low = df_htf['low'].values
    htf_close = df_htf['close'].values
    
    # Calculate CPR components
    tc = (htf_high + htf_low) / 2.0
    bc = (htf_high + htf_low + htf_close) / 3.0
    pivot = (tc + bc) / 2.0
    
    # CPR width (normalized by pivot level)
    cpr_width = np.abs(tc - bc) / np.where(pivot > 0, pivot, 1.0)
    
    # Align to 15m timeframe (shift by 1 to use completed HTF bar)
    tc_aligned = align_htf_to_ltf(prices, df_htf, tc)
    bc_aligned = align_htf_to_ltf(prices, df_htf, bc)
    pivot_aligned = align_htf_to_ltf(prices, df_htf, pivot)
    width_aligned = align_htf_to_ltf(prices, df_htf, cpr_width)
    
    return tc_aligned, bc_aligned, pivot_aligned, width_aligned

def get_session_hour(prices):
    """Extract UTC hour from open_time for session filtering"""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000.0
    hours = np.mod(np.floor(timestamps / 3600.0), 24).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA trend
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate CPR from 1d HTF
    tc_1d, bc_1d, pivot_1d, cpr_width_1d = calculate_cpr_from_htf(prices, df_1d)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # Session hours (UTC)
    session_hours = get_session_hour(prices)
    
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
    
    for i in range(100, n):
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
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(tc_1d[i]) or np.isnan(bc_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        in_session = session_hours[i] >= 0 and session_hours[i] <= 12
        
        # === CPR ANALYSIS ===
        cpr_narrow = cpr_width_1d[i] < 0.015  # <1.5% = narrow CPR (breakout potential)
        
        # Price position relative to CPR
        above_tc = close[i] > tc_1d[i]
        below_bc = close[i] < bc_1d[i]
        inside_cpr = close[i] >= bc_1d[i] and close[i] <= tc_1d[i]
        
        # === 1h TREND FILTER ===
        hma_1h_bull = close[i] > hma_1h_aligned[i]
        hma_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === ENTRY LOGIC (4-way confluence) ===
        desired_signal = 0.0
        
        # LONG: above TC + narrow CPR + 1h bull + RSI>50 + session
        if above_tc and cpr_narrow and hma_1h_bull and rsi_7[i] > 50.0 and in_session:
            # Stronger if RSI confirms momentum (55-70 range, not overbought)
            if rsi_7[i] > 55.0 and rsi_7[i] < 75.0:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: below BC + narrow CPR + 1h bear + RSI<50 + session
        elif below_bc and cpr_narrow and hma_1h_bear and rsi_7[i] < 50.0 and in_session:
            # Stronger if RSI confirms momentum (25-45 range, not oversold)
            if rsi_7[i] < 45.0 and rsi_7[i] > 20.0:
                desired_signal = -SIZE_STRONG
            else:
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