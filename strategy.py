#!/usr/bin/env python3
"""
Experiment #170: 1h Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 1h timeframe with strict confluence filters can achieve better Sharpe than 6h
by using proven regime detection (Choppiness Index) + mean reversion (cRSI) + HTF trend.

Key learnings from 169 experiments:
- 15m strategies fail with Sharpe=0.000 (ZERO trades) from overly strict conditions
- 6h marginally works but doesn't beat baseline (Sharpe=0.167)
- 4h strategies consistently fail (negative Sharpe)
- Session filter (08-20 UTC) improves quality by avoiding low-liquidity hours
- cRSI + Choppiness combination has research backing (75% win rate)

New approach for 1h:
- 4h HMA(21) for trend direction (HTF confirmation)
- 1d HMA(50) for major trend bias (regime filter)
- Choppiness Index(14) for regime detection (>61.8 = range, <38.2 = trend)
- Connors RSI (cRSI) for mean reversion entries (extremes <25/>75)
- Session filter: 08-20 UTC only (avoid Asian low-liquidity)
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.20 (conservative to control drawdown)

Design for trade generation (CRITICAL - avoid 0 trades):
- LOOSE cRSI thresholds (25/75 not 10/90)
- Multiple entry paths (primary + fallback)
- Fallback: enter when HTF strongly aligned (ignore some filters)
- Session filter ensures quality but doesn't block all trades
- Target 40-80 trades/year on 1h timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_session_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - detects ranging vs trending markets
    CHOP > 61.8 = ranging (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low < 1e-10:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of Streak Length (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    rsi_streak = calculate_rsi(streak_abs, streak_period)
    
    # PercentRank(100) - rank of today's change vs last 100 days
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        changes = np.diff(close[i-rank_period:i+1])
        if len(changes) > 0:
            today_change = changes[-1]
            rank = np.sum(changes[:-1] <= today_change) / (len(changes) - 1)
            percent_rank[i] = rank * 100.0
    
    # Combine
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(open_time_arr):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_arr // 1000) // 3600) % 24
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
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Extract hour for session filter
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
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
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
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Only trade during high-liquidity hours
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === WEEKLY REGIME (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP < 45 = trending (use trend entries)
        # CHOP > 55 = ranging (use mean reversion entries)
        is_trending = choppiness[i] < 45.0
        is_ranging = choppiness[i] > 55.0
        
        # === cRSI EXTREMES ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY LONG: Trending regime + HTF aligned + cRSI pullback
        if is_trending and htf_4h_bull and htf_1d_bull and crsi_oversold and above_sma200 and in_session:
            desired_signal = SIZE
        
        # PRIMARY SHORT: Trending regime + HTF aligned + cRSI rally
        elif is_trending and htf_4h_bear and htf_1d_bear and crsi_overbought and below_sma200 and in_session:
            desired_signal = -SIZE
        
        # FALLBACK 1: Ranging regime + mean reversion at extremes (ignore HTF)
        elif is_ranging and crsi[i] < 20.0 and in_session:
            desired_signal = SIZE * 0.7
        
        elif is_ranging and crsi[i] > 80.0 and in_session:
            desired_signal = -SIZE * 0.7
        
        # FALLBACK 2: Strong HTF alignment (ignore choppiness) - ensures trades
        elif htf_4h_bull and htf_1d_bull and crsi[i] < 30.0 and above_sma200 and in_session:
            desired_signal = SIZE * 0.6
        
        elif htf_4h_bear and htf_1d_bear and crsi[i] > 70.0 and below_sma200 and in_session:
            desired_signal = -SIZE * 0.6
        
        # FALLBACK 3: Very strong cRSI extreme (guarantee some trades)
        elif crsi[i] < 15.0 and in_session:
            desired_signal = SIZE * 0.5
        
        elif crsi[i] > 85.0 and in_session:
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
        if desired_signal >= SIZE * 0.95:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.95:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.65:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.65:
            final_signal = -SIZE * 0.7
        elif desired_signal >= SIZE * 0.45:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.45:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.25:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.25:
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