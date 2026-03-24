#!/usr/bin/env python3
"""
Experiment #905: 15m Primary + 4h/1d HTF — HMA Trend + Daily Pivot + RSI Pullback

Hypothesis: 15m timeframe with 4h trend bias and daily pivot levels provides
optimal intraday entry precision while maintaining HTF signal quality. Daily
CPR (Central Pivot Range) from 1d data gives institutional S/R levels. 15m RSI(7)
captures pullback entries within the HTF trend. Session filter (00-12 UTC)
reduces noise and trade count.

Key innovations:
1. 4h HMA(21) for primary trend direction - smooth, less whipsaw than EMA
2. 1d CPR (BC/TC/ Pivot) for institutional support/resistance levels
3. 15m RSI(7) for pullback entry timing within HTF trend
4. UTC session filter (00-12) - London/NY overlap, highest volume
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Entry conditions (LOOSE for trades, selective for quality):
- LONG: 4h HMA bull + price > Daily TC + 15m RSI(7) < 45 (pullback in uptrend)
- SHORT: 4h HMA bear + price < Daily BC + 15m RSI(7) > 55 (pullback in downtrend)
- Session filter: only 00-12 UTC (reduces trades by ~50%)

Target: Sharpe>0.45, trades>=40/year, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_daily_pivot_session_4h1d_v1"
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
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
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

def calculate_daily_pivot(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from 1d data
    Returns: pivot, bc (bottom), tc (top) arrays aligned to 1d bars
    
    Standard Pivot Formula:
    Pivot = (High + Low + Close) / 3
    BC = (High + Low) / 2
    TC = Pivot + (High - Low) / 2  (or Pivot - (High-Low)/2 for bottom)
    
    We use: BC = Pivot - range/2, TC = Pivot + range/2
    """
    n = len(df_1d)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    pivot = (high + low + close) / 3.0
    price_range = high - low
    bc = pivot - price_range / 4.0  # Bottom Central
    tc = pivot + price_range / 4.0  # Top Central
    
    return pivot, bc, tc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d, bc_1d, tc_1d = calculate_daily_pivot(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_aligned[i]) or np.isnan(bc_aligned[i]) or np.isnan(tc_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === 4h TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === DAILY CPR LEVELS ===
        price_above_tc = close[i] > tc_aligned[i]
        price_below_bc = close[i] < bc_aligned[i]
        price_in_cpr = (close[i] >= bc_aligned[i]) and (close[i] <= tc_aligned[i])
        
        # === RSI PULLBACK CONDITIONS (LOOSE for trades) ===
        rsi_oversold = rsi_7[i] < 45.0  # Pullback in uptrend
        rsi_overbought = rsi_7[i] > 55.0  # Pullback in downtrend
        rsi_extreme_long = rsi_7[i] < 35.0  # Strong long signal
        rsi_extreme_short = rsi_7[i] > 65.0  # Strong short signal
        
        # === ENTRY LOGIC (LOOSE ENOUGH FOR TRADES) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session:
            if htf_4h_bull and price_above_tc:
                # Bullish: trend up + above daily resistance (breakout)
                if rsi_oversold or rsi_extreme_long:
                    if rsi_extreme_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            elif htf_4h_bear and price_below_bc:
                # Bearish: trend down + below daily support (breakdown)
                if rsi_overbought or rsi_extreme_short:
                    if rsi_extreme_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            
            # Additional: mean reversion inside CPR when 4h trend unclear
            elif price_in_cpr:
                # Range-bound: fade CPR boundaries
                if rsi_extreme_long and close[i] < bc_aligned[i] * 1.002:
                    desired_signal = SIZE_BASE
                elif rsi_extreme_short and close[i] > tc_aligned[i] * 0.998:
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