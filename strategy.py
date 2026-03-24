#!/usr/bin/env python3
"""
Experiment #889: 15m Primary + 1h/1d HTF — Daily CPR Pivot + HMA Trend + RSI Entry

Hypothesis: 15m timeframe with daily pivot levels and 1h trend filter provides
optimal entry precision while maintaining HTF bias. Daily CPR (Central Pivot Range)
from institutional pivot methodology gives key support/resistance levels. 1h HMA
filters counter-trend trades. 15m RSI(7) provides fast entry timing.

Key innovations:
1. Daily CPR (BC/TC/Pivot) from 1d HTF - institutional pivot levels
2. 1h HMA(21) for intermediate trend bias
3. 15m RSI(7) for fast entry timing (oversold/overbought)
4. Session filter: 00-12 UTC (London/NY overlap for crypto)
5. CPR Width filter: narrow CPR = breakout day, wide CPR = range day
6. ATR(14) 2.5x trailing stop

Entry conditions:
- LONG: 1d bias bull + 1h HMA bull + price > TC + RSI(7) < 35 (pullback) + session
- SHORT: 1d bias bear + 1h HMA bear + price < BC + RSI(7) > 65 (rally) + session

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_pivot_hma_rsi_session_1h1d_v1"
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
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.dot(window, weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
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
    
    rsi = np.full(n, np.nan)
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

def calculate_daily_cpr_from_htf(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from 1d HTF data
    CPR = Pivot Range used by institutional traders
    
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = (Pivot - BC) + Pivot = 2*Pivot - BC
    
    Returns arrays aligned to 1d bars
    """
    n = len(df_1d)
    
    pivot = np.full(n, np.nan)
    bc = np.full(n, np.nan)
    tc = np.full(n, np.nan)
    cpr_width = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        bc[i] = (prev_high + prev_low) / 2.0
        tc[i] = 2 * pivot[i] - bc[i]
        
        # CPR Width as % of price (narrow = breakout potential)
        if pivot[i] > 1e-10:
            cpr_width[i] = abs(tc[i] - bc[i]) / pivot[i] * 100.0
        else:
            cpr_width[i] = 0.0
    
    return pivot, bc, tc, cpr_width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d CPR
    pivot_1d_raw, bc_1d_raw, tc_1d_raw, cpr_width_1d_raw = calculate_daily_cpr_from_htf(df_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    cpr_width_1d_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d_raw)
    
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
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(pivot_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        is_active_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === 1d CPR LEVELS ===
        pivot = pivot_1d_aligned[i]
        bc = bc_1d_aligned[i]
        tc = tc_1d_aligned[i]
        cpr_width = cpr_width_1d_aligned[i]
        
        # Narrow CPR = breakout day (< 0.5% of price)
        narrow_cpr = not np.isnan(cpr_width) and cpr_width < 0.5
        
        # === 1h HMA TREND BIAS ===
        hma_1h_bull = close[i] > hma_1h_aligned[i]
        hma_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === 1d PIVOT BIAS ===
        pivot_bull = close[i] > pivot
        pivot_bear = close[i] < pivot
        
        # === RSI CONDITIONS (15m fast RSI) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_neutral_long = rsi_7[i] < 45.0
        rsi_neutral_short = rsi_7[i] > 55.0
        
        # === PRICE VS CPR LEVELS ===
        price_above_tc = close[i] > tc
        price_below_bc = close[i] < bc
        price_in_cpr = close[i] >= bc and close[i] <= tc
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: 1h bull + 1d pivot bull + price > TC + RSI pullback + session
        long_confluence = 0
        if hma_1h_bull:
            long_confluence += 1
        if pivot_bull:
            long_confluence += 1
        if price_above_tc:
            long_confluence += 1
        if rsi_oversold or rsi_neutral_long:
            long_confluence += 1
        if is_active_session:
            long_confluence += 1
        
        # SHORT: 1h bear + 1d pivot bear + price < BC + RSI rally + session
        short_confluence = 0
        if hma_1h_bear:
            short_confluence += 1
        if pivot_bear:
            short_confluence += 1
        if price_below_bc:
            short_confluence += 1
        if rsi_overbought or rsi_neutral_short:
            short_confluence += 1
        if is_active_session:
            short_confluence += 1
        
        # Require 4+ confluence for entry (out of 5 factors)
        if long_confluence >= 4:
            if rsi_oversold:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        if short_confluence >= 4:
            if rsi_overbought:
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