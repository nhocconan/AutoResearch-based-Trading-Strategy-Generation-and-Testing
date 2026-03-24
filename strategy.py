#!/usr/bin/env python3
"""
Experiment #003: 6h Primary + 1d/1w HTF — Weekly Pivot Bounce + Daily Trend + EMA Pullback

Hypothesis: 6h timeframe is unexplored middle ground between 4h (trend) and 12h (swing).
Key edge: Weekly pivot levels (R1, S1, P) provide major S/R that 6h candles respect.
Combined with 1d HMA trend bias + 6h EMA21 pullback entries = high-probability setups.

Why this should work:
- Weekly pivots are INSTITUTIONAL levels (banks, funds watch these)
- 6h pullback to EMA21 in uptrend = classic continuation pattern
- Choppiness filter avoids entering during range-bound periods
- LOOSE RSI filter (25-75) ensures trades generate on all symbols
- 1d HMA provides major trend without being too restrictive

Design:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1w pivots (major S/R), 1d HMA (trend bias)
- Entry: 6h EMA21 pullback + RSI filter + Choppiness regime
- Position size: 0.28 (28% of capital)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_ema_pullback_1d1w_v1"
timeframe = "6h"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_weekly_pivots(df_1w):
    """
    Calculate weekly pivot levels from 1w OHLC
    Classic Pivot: P = (H + L + C) / 3
    R1 = 2*P - L, S1 = 2*P - H
    R2 = P + (H - L), S2 = P - (H - L)
    """
    n = len(df_1w)
    pivots = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    
    for i in range(n):
        h = df_1w['high'].iloc[i]
        l = df_1w['low'].iloc[i]
        c = df_1w['close'].iloc[i]
        
        p = (h + l + c) / 3.0
        pivots[i] = p
        r1[i] = 2.0 * p - l
        s1[i] = 2.0 * p - h
    
    return pivots, r1, s1

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w pivot levels
    pivot_1w, r1_1w, s1_1w = calculate_weekly_pivots(df_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Calculate primary (6h) indicators
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
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
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY PIVOT SUPPORT/RESISTANCE ===
        # Price near weekly S1 (support) or R1 (resistance)
        pivot_range = r1_aligned[i] - s1_aligned[i]
        if pivot_range < 1e-10:
            pivot_range = close[i] * 0.05
        
        near_s1 = abs(close[i] - s1_aligned[i]) < pivot_range * 0.15
        near_r1 = abs(close[i] - r1_aligned[i]) < pivot_range * 0.15
        near_pivot = abs(close[i] - pivot_aligned[i]) < pivot_range * 0.10
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === EMA PULLBACK SETUP ===
        # Long: price pulls back to EMA21 in uptrend
        # Short: price rallies to EMA21 in downtrend
        ema_bull = ema_21[i] > ema_50[i] and close[i] > ema_21[i]
        ema_bear = ema_21[i] < ema_50[i] and close[i] < ema_21[i]
        
        # Price touching/near EMA21 (pullback entry)
        ema_touch_long = close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99
        ema_touch_short = close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.01
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 25.0 and rsi[i] < 75.0
        rsi_ok_short = rsi[i] > 25.0 and rsi[i] < 75.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === DESIRED SIGNAL (Multi-confluence logic) ===
        desired_signal = 0.0
        
        # LONG setups (multiple confluence requirements, but LOOSE enough)
        long_score = 0
        
        # Confluence 1: HTF trend bullish
        if htf_bull:
            long_score += 1
        
        # Confluence 2: EMA structure bullish
        if ema_bull:
            long_score += 1
        
        # Confluence 3: Price at/near EMA21 (pullback)
        if ema_touch_long:
            long_score += 1
        
        # Confluence 4: Near weekly support (S1 or pivot)
        if near_s1 or near_pivot:
            long_score += 1
        
        # Confluence 5: RSI not overbought
        if rsi_ok_long:
            long_score += 1
        
        # Confluence 6: Not too choppy (or choppy but at support)
        if is_trending or (is_choppy and (near_s1 or near_pivot)):
            long_score += 1
        
        # SHORT setups
        short_score = 0
        
        if htf_bear:
            short_score += 1
        if ema_bear:
            short_score += 1
        if ema_touch_short:
            short_score += 1
        if near_r1 or near_pivot:
            short_score += 1
        if rsi_ok_short:
            short_score += 1
        if is_trending or (is_choppy and (near_r1 or near_pivot)):
            short_score += 1
        
        # Entry thresholds (LOOSE to ensure trades)
        if long_score >= 3 and rsi[i] < 70.0:
            desired_signal = SIZE
        elif short_score >= 3 and rsi[i] > 30.0:
            desired_signal = -SIZE
        # Fallback: strong HTF trend + EMA cross
        elif htf_bull and ema_bull and rsi[i] < 65.0:
            desired_signal = SIZE * 0.7
        elif htf_bear and ema_bear and rsi[i] > 35.0:
            desired_signal = -SIZE * 0.7
        # Fallback 2: weekly pivot bounce
        elif near_s1 and rsi_oversold and hma_1d_aligned[i] < close[i]:
            desired_signal = SIZE * 0.7
        elif near_r1 and rsi_overbought and hma_1d_aligned[i] > close[i]:
            desired_signal = -SIZE * 0.7
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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