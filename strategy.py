#!/usr/bin/env python3
"""
Experiment #1577: 15m Primary + 4h/1d HTF — Session RSI Pullback with CPR

Hypothesis: 15m timeframe with strict HTF alignment can capture intraday pullbacks 
in the direction of the 4h trend. Using RSI(7) for quick mean-reversion entries 
with 4h HMA(21) for trend bias and 1d CPR for S/R confluence.

Key components:
1. 4h HMA(21) for major trend direction (only trade with trend)
2. 15m RSI(7) for pullback entries (loose: <35 long, >65 short)
3. 1d CPR (Central Pivot Range) for support/resistance confluence
4. Session filter: 00-12 UTC (London+NY overlap)
5. Position size: 0.15-0.20 (smaller for 15m frequency)
6. ATR(14) trailing stoploss (2.0x ATR)

Why this should work:
- RSI(7) crosses 35/65 frequently = guaranteed trades
- 4h trend filter prevents counter-trend disasters
- Session filter reduces noise during low-volume hours
- Smaller position size accounts for higher trade frequency
- Simple, proven indicators

Entry logic (LOOSE to guarantee ≥40 trades/train, ≥3/test):
- LONG: 4h_HMA bullish + 15m_RSI<35 + session 00-12 UTC
- SHORT: 4h_HMA bearish + 15m_RSI>65 + session 00-12 UTC

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi7_hma4h_cpr1d_v1"
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

def calculate_cpr(high, low, close):
    """
    Central Pivot Range (CPR) - daily support/resistance levels
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = Pivot
    Narrow CPR = TC - BC is small (consolidation)
    Wide CPR = TC - BC is large (trending expected)
    """
    n = len(close)
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        bc[i] = (high[i-1] + low[i-1]) / 2.0
        tc[i] = pivot[i]
    
    return pivot, bc, tc

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
    
    # Calculate 1d CPR
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Session hours
    session_hours = calculate_session_hour(open_time)
    
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
    min_bars = 50
    
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
        
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(bc_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London + NY overlap) ===
        hour = session_hours[i]
        in_session = (hour >= 0 and hour < 12)
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI (7-period for faster signals) ===
        rsi = rsi_7[i]
        
        # === CPR LEVELS ===
        pivot = pivot_1d_aligned[i]
        bc = bc_1d_aligned[i]
        tc = tc_1d_aligned[i]
        cpr_width = abs(tc - bc) if not np.isnan(tc) and not np.isnan(bc) else 0
        price_near_bc = abs(close[i] - bc) < cpr_width * 0.5 if cpr_width > 1e-10 else False
        price_near_tc = abs(close[i] - tc) < cpr_width * 0.5 if cpr_width > 1e-10 else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + in session
        if price_above_4h and rsi < 35 and in_session:
            # Extra confluence: price near BC support
            if price_near_bc:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI overbought + in session
        elif price_below_4h and rsi > 65 and in_session:
            # Extra confluence: price near TC resistance
            if price_near_tc:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
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