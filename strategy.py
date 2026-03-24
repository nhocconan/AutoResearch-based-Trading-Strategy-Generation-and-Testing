#!/usr/bin/env python3
"""
Experiment #020: 6h Primary + 1d/1w HTF — Weekly Pivot + Connors RSI + Choppiness Regime

Hypothesis: 6h timeframe is underexplored (only 3 attempts, all failed). This strategy combines:
- Weekly Pivot levels (traditional S/R that institutions watch)
- Connors RSI (CRSI) - proven 75% win rate mean reversion indicator
- Choppiness Index regime filter (trend vs range)
- 1d HMA for major trend bias

Why this might work:
1. Weekly pivots are REAL levels traders watch (not synthetic indicators)
2. CRSI catches oversold/overbought better than standard RSI
3. CHOP regime prevents trend-following in choppy markets
4. 6h gives enough bars for signal generation without fee drag of lower TF

Target: 30-60 trades/year, Sharpe>0.35, DD>-40%, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_crsi_chop_1d1w_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Very fast RSI on price
    RSI(streak, 2): RSI on consecutive up/down days
    PercentRank(100): Percentile rank of today's change over last 100 days
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_abs = np.abs(streak)
    # Convert streak to gains/losses for RSI calculation
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank(100) - percentile of today's change over last 100 days
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        today_change = close[i] - close[i-1] if i > 0 else 0
        past_changes = np.diff(close[i-rank_period+1:i+1])
        count_below = np.sum(past_changes < today_change)
        percent_rank[i] = 100.0 * count_below / len(past_changes) if len(past_changes) > 0 else 50.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    Calculate Weekly Pivot levels from 1w OHLC data
    P = (H + L + C) / 3
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    """
    n = len(df_1w)
    pivots = np.zeros((n, 5))  # P, R1, R2, S1, S2
    pivots[:] = np.nan
    
    for i in range(n):
        h = df_1w['high'].iloc[i]
        l = df_1w['low'].iloc[i]
        c = df_1w['close'].iloc[i]
        
        p = (h + l + c) / 3.0
        r1 = 2.0 * p - l
        s1 = 2.0 * p - h
        r2 = p + (h - l)
        s2 = p - (h - l)
        
        pivots[i] = [p, r1, r2, s1, s2]
    
    return pivots

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
    
    # Calculate and align weekly pivots
    weekly_pivots_raw = calculate_weekly_pivots(df_1w)
    # Align each pivot level separately
    pivot_p_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 0])
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 1])
    pivot_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 2])
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 3])
    pivot_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivots_raw[:, 4])
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(crsi[i]):
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(pivot_p_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY PIVOT LEVELS ===
        pivot_p = pivot_p_aligned[i]
        pivot_s1 = pivot_s1_aligned[i]
        pivot_s2 = pivot_s2_aligned[i]
        pivot_r1 = pivot_r1_aligned[i]
        pivot_r2 = pivot_r2_aligned[i]
        
        # Distance to pivot levels (normalized)
        dist_to_p = abs(close[i] - pivot_p) / (pivot_p + 1e-10)
        dist_to_s1 = abs(close[i] - pivot_s1) / (pivot_s1 + 1e-10)
        dist_to_s2 = abs(close[i] - pivot_s2) / (pivot_s2 + 1e-10)
        dist_to_r1 = abs(close[i] - pivot_r1) / (pivot_r1 + 1e-10)
        dist_to_r2 = abs(close[i] - pivot_r2) / (pivot_r2 + 1e-10)
        
        # Near support (for long entries)
        near_s1 = dist_to_s1 < 0.015  # within 1.5% of S1
        near_s2 = dist_to_s2 < 0.015  # within 1.5% of S2
        near_support = near_s1 or near_s2
        
        # Near resistance (for short entries)
        near_r1 = dist_to_r1 < 0.015  # within 1.5% of R1
        near_r2 = dist_to_r2 < 0.015  # within 1.5% of R2
        near_resistance = near_r1 or near_r2
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # extremely oversold
        crsi_overbought = crsi[i] > 85.0  # extremely overbought
        crsi_neutral_oversold = crsi[i] < 30.0
        crsi_neutral_overbought = crsi[i] > 70.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL (Dual Regime + Pivot Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow HTF bias with CRSI confirmation
            # LONG: HTF bull + CRSI oversold + near support or HMA bull
            if htf_bull and crsi_neutral_oversold and (near_support or hma_bull):
                desired_signal = SIZE
            # SHORT: HTF bear + CRSI overbought + near resistance or HMA bear
            elif htf_bear and crsi_neutral_overbought and (near_resistance or hma_bear):
                desired_signal = -SIZE
            # Fallback: extreme CRSI mean reversion with HTF not against
            elif crsi_oversold and not htf_bear:
                desired_signal = SIZE * 0.6
            elif crsi_overbought and not htf_bull:
                desired_signal = -SIZE * 0.6
        else:
            # CHOPPY REGIME: Mean revert at pivot levels with CRSI extreme
            # LONG: near support + CRSI oversold
            if near_support and crsi_oversold:
                desired_signal = SIZE
            # SHORT: near resistance + CRSI overbought
            elif near_resistance and crsi_overbought:
                desired_signal = -SIZE
            # Fallback: extreme CRSI alone
            elif crsi_oversold and hma_bull:
                desired_signal = SIZE * 0.6
            elif crsi_overbought and hma_bear:
                desired_signal = -SIZE * 0.6
        
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