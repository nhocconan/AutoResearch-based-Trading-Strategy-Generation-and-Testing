#!/usr/bin/env python3
"""
Experiment #1351: 4h Primary + 1d/1w HTF — Adaptive Trend Following with KAMA

Hypothesis: 4h strategies fail due to rigid trend filters (HMA/EMA) that whipsaw in 
range markets. KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast 
in trends, slow in chop. Combined with 1d/1w HTF for macro bias, this should:
1. Reduce whipsaws in 2022 crash and 2025 bear market
2. Capture trends when ADX confirms strength
3. Generate 20-50 trades/year (4h target) with better win rate

Key design choices:
1. KAMA(10,2,30) on 4h - adapts to market regime automatically
2. 1d KAMA(21) for intermediate trend - confirms 4h signals
3. 1w KAMA(21) for macro regime - only trade with weekly trend
4. ADX(14) > 20 for trend strength - not too strict (>30 kills trades)
5. RSI(14) 40-60 bands - entry timing without over-filtering
6. ATR(14) trailing stop 2.5x - proven risk management
7. Position size 0.30 - discrete levels for fee efficiency

Target: 25-45 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_adx_rsi_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - fast in trends, slow in chop
    period: efficiency ratio lookback
    fast_period: fastest smoothing constant (2/(fast+1))
    slow_period: slowest smoothing constant (2/(slow+1))
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            signal = abs(close[i] - close[i - period])
            noise = 0.0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    noise += abs(close[j] - close[j - 1])
            if noise > 1e-10:
                er[i] = signal / noise
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA
    kama[period] = np.nanmean(close[:period + 1])
    
    # Calculate adaptive KAMA
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]) and not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = range/chop
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's method
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    tr_smooth[period - 1] = np.sum(tr[:period])
    plus_dm_smooth[period - 1] = np.sum(plus_dm[:period])
    minus_dm_smooth[period - 1] = np.sum(minus_dm[:period])
    
    for i in range(period, n):
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
    
    # Calculate DI and DX
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period - 1:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx

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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA for trend filters
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21, fast_period=2, slow_period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size - discrete
    
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
            continue
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME (1w KAMA) ===
        # Only trade in direction of weekly trend
        macro_bull = close[i] > kama_1w_aligned[i]
        macro_bear = close[i] < kama_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d KAMA) ===
        trend_1d_bull = close[i] > kama_1d_aligned[i]
        trend_1d_bear = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        trend_4h_bull = close[i] > kama_4h[i]
        trend_4h_bear = close[i] < kama_4h[i]
        
        # === TREND STRENGTH (ADX) ===
        # ADX > 20 = trending, ADX < 20 = chop (but still allow trades)
        trend_strong = adx[i] > 20.0
        trend_very_strong = adx[i] > 25.0
        
        # === RSI MOMENTUM (moderate bands for trade frequency) ===
        rsi_bull = rsi[i] > 45.0
        rsi_bear = rsi[i] < 55.0
        rsi_neutral = 40.0 < rsi[i] < 60.0
        
        # === KAMA SLOPE (trend confirmation) ===
        kama_slope_bull = False
        kama_slope_bear = False
        if i >= 5 and not np.isnan(kama_4h[i - 5]):
            kama_slope_bull = kama_4h[i] > kama_4h[i - 5]
            kama_slope_bear = kama_4h[i] < kama_4h[i - 5]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple paths to ensure trades happen
        if macro_bull:  # Weekly trend is bull - primary filter
            # Path 1: All trends align + ADX confirms (strongest signal)
            if trend_1d_bull and trend_4h_bull and trend_strong and rsi_bull:
                desired_signal = BASE_SIZE
            # Path 2: 4h and 1d align + KAMA slope up (good signal)
            elif trend_4h_bull and trend_1d_bull and kama_slope_bull:
                desired_signal = BASE_SIZE
            # Path 3: 4h bull + RSI confirmation (weaker but allows trades)
            elif trend_4h_bull and rsi_bull and kama_slope_bull:
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY: Multiple paths to ensure trades happen
        elif macro_bear:  # Weekly trend is bear - primary filter
            # Path 1: All trends align + ADX confirms (strongest signal)
            if trend_1d_bear and trend_4h_bear and trend_strong and rsi_bear:
                desired_signal = -BASE_SIZE
            # Path 2: 4h and 1d align + KAMA slope down (good signal)
            elif trend_4h_bear and trend_1d_bear and kama_slope_bear:
                desired_signal = -BASE_SIZE
            # Path 3: 4h bear + RSI confirmation (weaker but allows trades)
            elif trend_4h_bear and rsi_bear and kama_slope_bear:
                desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
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
                # Flip position
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