#!/usr/bin/env python3
"""
Experiment #265: 15m Primary + 4h/1d HTF — Daily Pivot + RSI Mean Reversion v1

Hypothesis: 15m timeframe can work with VERY selective entries using daily pivot levels
as key S/R zones. This combines:
1. Daily pivot levels (from 1d HTF) as support/resistance zones
2. 4h HMA(21) for intermediate trend direction
3. 15m RSI(7) for oversold/overbought entry timing
4. Session filter: 00-12 UTC (London+NY overlap, highest crypto volume)
5. Require 3+ confluence before entry

Key insight from failed 15m experiments (#257, #261):
- Too many trades = fee death. Must be VERY selective.
- Use HTF for DIRECTION, 15m only for ENTRY TIMING
- Position size smaller (0.15-0.25) due to higher frequency

Daily Pivot Calculation (Standard):
P = (H + L + C) / 3
R1 = 2*P - L, S1 = 2*P - H
R2 = P + (H - L), S2 = P - (H - L)
TC (Top Central) = (H + L) / 2, BC (Bottom Central) = (H + L + C) / 3

Entry Logic:
- Long: price near S1/S2 + RSI(7)<30 + 4h HMA bullish + session filter
- Short: price near R1/R2 + RSI(7)>70 + 4h HMA bearish + session filter

Position sizing: 0.15 base, 0.25 strong (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 40-100 trades/year, Sharpe>0.40, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_daily_pivot_rsi_session_4h1d_v1"
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

def calculate_daily_pivots(df_1d):
    """
    Calculate daily pivot levels from 1d data.
    Returns arrays aligned to 1d bars that will be aligned to 15m later.
    """
    n = len(df_1d)
    
    # Pivot = (H + L + C) / 3
    pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2.0 * pivot - df_1d['low'].values
    s1 = 2.0 * pivot - df_1d['high'].values
    
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pivot + (df_1d['high'].values - df_1d['low'].values)
    s2 = pivot - (df_1d['high'].values - df_1d['low'].values)
    
    # TC (Top Central) = (H + L) / 2, BC (Bottom Central) = Pivot
    tc = (df_1d['high'].values + df_1d['low'].values) / 2.0
    bc = pivot
    
    return pivot, r1, s1, r2, s2, tc, bc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate daily pivot levels from 1d data
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d, tc_1d, bc_1d = calculate_daily_pivots(df_1d)
    
    # Align daily pivots to 15m (use previous day's pivots = shift by 1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === 4H TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DAILY PIVOT ZONES ===
        # Check if price is near support (S1, S2) or resistance (R1, R2)
        # Use 0.5% tolerance for "near"
        tol = 0.005
        
        near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < tol
        near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < tol
        near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < tol
        near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < tol
        
        near_support = near_s1 or near_s2
        near_resistance = near_r1 or near_r2
        
        # Also check if price is BETWEEN pivot levels (value area)
        in_value_area_long = (close[i] > s1_aligned[i]) and (close[i] < pivot_aligned[i])
        in_value_area_short = (close[i] < r1_aligned[i]) and (close[i] > pivot_aligned[i])
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Near support + RSI oversold + 4h bullish + session filter
        # Need 3+ confluence
        long_confluence = 0
        if near_support:
            long_confluence += 1
        if rsi_oversold:
            long_confluence += 1
        if htf_4h_bull:
            long_confluence += 1
        if above_sma200:
            long_confluence += 1
        if in_session:
            long_confluence += 0.5  # session is soft filter
        
        # SHORT: Near resistance + RSI overbought + 4h bearish + session filter
        short_confluence = 0
        if near_resistance:
            short_confluence += 1
        if rsi_overbought:
            short_confluence += 1
        if htf_4h_bear:
            short_confluence += 1
        if below_sma200:
            short_confluence += 1
        if in_session:
            short_confluence += 0.5
        
        # Require 3+ confluence for entry
        if long_confluence >= 3.0 and rsi_oversold:
            # Strong signal if 4h trend + SMA200 aligned
            if htf_4h_bull and above_sma200:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        if short_confluence >= 3.0 and rsi_overbought:
            if htf_4h_bear and below_sma200:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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