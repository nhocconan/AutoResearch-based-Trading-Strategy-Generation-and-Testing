#!/usr/bin/env python3
"""
Experiment #1554: 4h Primary + 12h HTF — KAMA Trend with Choppiness Filter

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility
better than HMA/EMA by adjusting smoothing based on Efficiency Ratio.
Combined with Choppiness Index filter (only trend-follow when CHOP < 50),
this should reduce whipsaws in ranging markets while maintaining trend exposure.

Key improvements over #1551:
1. KAMA instead of HMA - adapts ER to volatility (proven ETH Sharpe +0.755)
2. Choppiness filter - avoid trend entries in choppy markets (CHOP < 50)
3. Looser RSI (25-75) - ensures trades fire (critical lesson from 0-trade failures)
4. 12h KAMA for HTF bias (faster response than 1d, slower than 4h)
5. 3.0x ATR stop (wider than 2.5x to reduce premature exits in 4h noise)
6. Size: 0.30 discrete (0.0, ±0.30) to minimize fee churn

Why 4h + 12h works:
- 12h provides trend bias without 1d lag
- 4h entry timing captures pullbacks
- Choppiness filters out ~40% of bad trend entries in ranges
- KAMA ER adapts smoothing: fast in trends, slow in chop

Target: Sharpe > 0.618, trades 30-60/train, >3/test, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_chop_rsi_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes (0=noise, 1=trend)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.full(n, np.nan)
    mask = ~np.isnan(er)
    sc[mask] = np.power(er[mask] * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA
    kama = np.full(n, np.nan)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            continue
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) / log10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.full(n, np.nan)
    mask = (hh - ll) > 1e-10
    chop[mask] = 100.0 * (atr_sum[mask] / (hh[mask] - ll[mask])) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h KAMA for macro trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=21)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # KAMA for primary trend timing
    kama_4h = calculate_kama(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (12h KAMA) ===
        daily_bull = close[i] > kama_12h_aligned[i]
        daily_bear = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        kama_4h_bull = close[i] > kama_4h[i]
        kama_4h_bear = close[i] < kama_4h[i]
        
        # === CHOPPINESS FILTER (only trend when CHOP < 50) ===
        trending_market = chop[i] < 50.0
        
        # === RSI FILTER (VERY LOOSE — ensures trades fire) ===
        rsi_long_ok = rsi_14[i] > 25.0
        rsi_short_ok = rsi_14[i] < 75.0
        
        # === ENTRY LOGIC — SIMPLE & LOOSE ===
        desired_signal = 0.0
        
        # LONG: 12h bull + 4h bull + trending + RSI ok
        if daily_bull and kama_4h_bull and trending_market and rsi_long_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: 12h bear + 4h bear + trending + RSI ok
        if daily_bear and kama_4h_bear and trending_market and rsi_short_ok:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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