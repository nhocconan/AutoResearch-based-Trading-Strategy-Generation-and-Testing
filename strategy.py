#!/usr/bin/env python3
"""
Experiment #679: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + HTF HMA

Hypothesis: 1h timeframe with regime-adaptive logic should handle both trending (2021)
and choppy/bear (2022-2025) periods. Key innovation: Choppiness Index determines whether
to use trend-following OR mean-reversion logic. Connors RSI provides high-probability
entry timing (75% win rate documented). 4h/12h HMA provides HTF bias filter.

Why this should work:
1. Choppiness(14) > 61.8 = range regime → use CRSI mean reversion (buy low, sell high)
2. Choppiness(14) < 38.2 = trend regime → use HTF HMA direction + pullback entries
3. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — captures oversold/overbought
4. Session filter 08-20 UTC reduces Asian session noise
5. 4h HMA(21) + 12h HMA(48) dual HTF confirmation

Entry conditions (LOOSE to ensure trades):
- RANGE: CHOP>61.8 + CRSI<15 → long, CRSI>85 → short
- TREND: CHOP<38.2 + price>4h_HMA + CRSI<40 → long, price<4h_HMA + CRSI>60 → short
- Size: 0.25 base, 0.30 with dual HTF confirmation

Target: Sharpe>0.40, trades>=40/year, DD<-30%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Fast momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Where current return ranks vs last 100 bars
    
    CRSI < 10 = extremely oversold (buy)
    CRSI > 90 = extremely overbought (sell)
    """
    n = len(close)
    if n < rank_period + rsi_period + streak_period:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    rsi_close[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_close[i] = 100.0
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period + 1, n):
        if streak_avg_loss[i] > 1e-10:
            rs = streak_avg_gain[i] / streak_avg_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        elif streak_avg_gain[i] > 1e-10:
            rsi_streak[i] = 100.0
        else:
            rsi_streak[i] = 50.0
    
    # Component 3: Percent Rank of returns over 100 periods
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
        else:
            returns[i] = 0.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            count_below = np.sum(valid < returns[i])
            percent_rank[i] = 100.0 * count_below / len(valid)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def get_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = get_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = choppiness[i] > 61.8
        is_trend = choppiness[i] < 38.2
        # Neutral zone: 38.2 <= CHOP <= 61.8
        
        # === HTF BIAS ===
        htf_bull_4h = close[i] > hma_4h_aligned[i]
        htf_bear_4h = close[i] < hma_4h_aligned[i]
        htf_bull_12h = close[i] > hma_12h_aligned[i]
        htf_bear_12h = close[i] < hma_12h_aligned[i]
        
        # Dual HTF confirmation
        htf_strong_bull = htf_bull_4h and htf_bull_12h
        htf_strong_bear = htf_bear_4h and htf_bear_12h
        
        # === ENTRY LOGIC (REGIME ADAPTIVE) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with CRSI extremes
        if is_range:
            if crsi[i] < 15:  # Extremely oversold
                if in_session:
                    desired_signal = SIZE_STRONG if htf_strong_bull else SIZE_BASE
            elif crsi[i] > 85:  # Extremely overbought
                if in_session:
                    desired_signal = -SIZE_STRONG if htf_strong_bear else -SIZE_BASE
        
        # TREND REGIME: Follow HTF direction with CRSI pullback
        elif is_trend:
            # Long: HTF bull + CRSI pullback (not overbought)
            if htf_bull_4h and crsi[i] < 45:
                if in_session:
                    desired_signal = SIZE_STRONG if htf_strong_bull else SIZE_BASE
            # Short: HTF bear + CRSI pullback (not oversold)
            elif htf_bear_4h and crsi[i] > 55:
                if in_session:
                    desired_signal = -SIZE_STRONG if htf_strong_bear else -SIZE_BASE
        
        # NEUTRAL REGIME: Only trade extreme CRSI with HTF confirmation
        else:
            if crsi[i] < 10 and htf_strong_bull and in_session:
                desired_signal = SIZE_BASE
            elif crsi[i] > 90 and htf_strong_bear and in_session:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
                entry_atr = atr[i]
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