#!/usr/bin/env python3
"""
Experiment #150: 1h Primary + 4h/1d HTF — cRSI Mean Reversion + Choppiness + Session Filter

Hypothesis: 1h timeframe with 4h/1d HTF trend bias + Connors RSI mean reversion entries
will generate 40-80 trades/year with positive Sharpe. This combines:
- 4h HMA(21) for intermediate trend direction
- 1d HMA(50) for major trend bias
- Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 for precise entries
- Choppiness Index(14) < 61.8 to avoid ranging markets
- Session filter 08-20 UTC to avoid low-volume periods
- Position size: 0.20 (conservative to control drawdown)

Why this might work:
- cRSI has 75% win rate in backtests (research note)
- CHOP filter prevents whipsaw in choppy markets
- HTF trend alignment reduces false signals
- Session filter avoids Asian low-volume chop
- 1h TF gives more entry opportunities than 4h/6h but fewer than 15m/30m

Design for trade generation (CRITICAL - avoid 0 trades):
- cRSI thresholds: <15 long, >85 short (wide enough to trigger)
- Fallback: enter on cRSI <25 or >75 if HTF trend very strong
- Target: 50-80 trades/year on 1h timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_session_4h1d_v1"
timeframe = "1h"
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

def calculate_rsi_streak(close, period=2):
    """
    Connors RSI Streak Component
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                streak = 1
                j = i - 1
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                streak = -1
                j = i - 1
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        
        # Convert streak to 0-100 scale (inverted for mean reversion)
        # Long streak up = high value, but we want low for long entry
        streak_abs = abs(streak)
        if streak_abs > period:
            streak_abs = period
        
        if streak >= 0:
            streak_rsi[i] = 100.0 * (1.0 - streak_abs / period)
        else:
            streak_rsi[i] = 100.0 * (streak_abs / period)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank: percentage of past prices lower than current
    For Connors RSI
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        count_lower = np.sum(window[:-1] < window[-1])
        pr[i] = 100.0 * count_lower / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme values (<10 or >90) indicate mean reversion opportunities
    """
    rsi3 = calculate_rsi(close, period=rsi_period)
    streak = calculate_rsi_streak(close, period=streak_period)
    pr = calculate_percent_rank(close, period=pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(max(rsi_period, streak_period, pr_period), n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi3[i] + streak[i] + pr[i]) / 3.0
    
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
    We use CHOP < 61.8 as filter to avoid choppy markets
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

def get_hour_from_open_time(open_time_col):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_col // 1000) // 3600) % 24
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
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative)
    
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Only trade during high-volume hours
        session_ok = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === CHOPPINESS FILTER ===
        # Only trade when CHOP < 61.8 (trending, not choppy)
        chop_ok = chop[i] < 61.8
        
        # === HTF BIAS (4h and 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === cRSI MEAN REVERSION SIGNALS ===
        # Extreme oversold (long) or overbought (short)
        crsi_extreme_long = crsi[i] < 15.0
        crsi_extreme_short = crsi[i] > 85.0
        
        # Moderate extremes (fallback)
        crsi_moderate_long = crsi[i] < 25.0
        crsi_moderate_short = crsi[i] > 75.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: cRSI extreme + HTF aligned + session + chop filter
        if crsi_extreme_long and htf_strong_bull and session_ok and chop_ok:
            desired_signal = SIZE
        
        elif crsi_extreme_short and htf_strong_bear and session_ok and chop_ok:
            desired_signal = -SIZE
        
        # FALLBACK 1: cRSI extreme + one HTF aligned (70% size)
        elif crsi_extreme_long and htf_4h_bull and session_ok and chop_ok:
            desired_signal = SIZE * 0.7
        
        elif crsi_extreme_short and htf_4h_bear and session_ok and chop_ok:
            desired_signal = -SIZE * 0.7
        
        # FALLBACK 2: cRSI moderate + strong HTF + session (50% size)
        elif crsi_moderate_long and htf_strong_bull and session_ok:
            desired_signal = SIZE * 0.5
        
        elif crsi_moderate_short and htf_strong_bear and session_ok:
            desired_signal = -SIZE * 0.5
        
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
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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