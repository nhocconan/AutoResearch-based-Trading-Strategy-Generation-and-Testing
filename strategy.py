#!/usr/bin/env python3
"""
Experiment #921: 15m Primary + 1h/4h/1d HTF — Daily CPR + HMA Trend + RSI Pullback

Hypothesis: 15m timeframe with strict multi-confluence filters can capture intraday 
momentum while avoiding fee drag. Daily CPR provides key support/resistance levels. 
4h HMA gives trend bias. 15m RSI(7) for precise entry timing. Session filter limits 
trades to high-liquidity periods (00-12 UTC).

Key innovations:
1. Daily CPR (Central Pivot Range) from 1d HTF - BC/TC as key levels
2. 4h HMA(21) for trend bias - price above = bullish, below = bearish
3. 15m RSI(7) for entry timing - oversold bounce in uptrend, overbought fade in downtrend
4. Session filter: only trade 00-12 UTC (London/NY overlap)
5. ATR(14) 2.5x trailing stop
6. Discrete sizing: 0.15-0.20 (smaller for 15m frequency)
7. 3+ confluence required for entry (HTF trend + CPR level + RSI + session)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%, trades/year <100
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_hma_rsi_session_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
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
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_daily_cpr(high, low, close):
    """
    Calculate Daily Central Pivot Range (CPR)
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = Pivot + (High - Low) / 2
    
    Uses current bar's OHLC - align_htf_to_ltf will handle the shift
    """
    n = len(close)
    pivot = np.full(n, np.nan)
    bc = np.full(n, np.nan)
    tc = np.full(n, np.nan)
    
    for i in range(1, n):
        pivot[i] = (high[i] + low[i] + close[i]) / 3.0
        bc[i] = (high[i] + low[i]) / 2.0
        tc[i] = pivot[i] + (high[i] - low[i]) / 2.0
    
    return pivot, bc, tc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate Daily CPR from 1d data
    pivot_1d, bc_1d, tc_1d = calculate_daily_cpr(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
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
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CPR POSITION ===
        cpr_width = (tc_aligned[i] - bc_aligned[i]) / pivot_aligned[i] if pivot_aligned[i] > 0 else 1.0
        narrow_cpr = cpr_width < 0.015
        
        above_cpr = close[i] > tc_aligned[i]
        below_cpr = close[i] < bc_aligned[i]
        inside_cpr = bc_aligned[i] <= close[i] <= tc_aligned[i]
        
        # === RSI SIGNALS (tighter for selectivity) ===
        rsi_oversold = rsi_7[i] < 28.0
        rsi_overbought = rsi_7[i] > 72.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + session + (above CPR + oversold OR inside CPR + oversold + narrow)
        if htf_4h_bull and in_session:
            if above_cpr and rsi_oversold:
                desired_signal = SIZE_STRONG
            elif inside_cpr and rsi_oversold and narrow_cpr:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + session + (below CPR + overbought OR inside CPR + overbought + narrow)
        elif htf_4h_bear and in_session:
            if below_cpr and rsi_overbought:
                desired_signal = -SIZE_STRONG
            elif inside_cpr and rsi_overbought and narrow_cpr:
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