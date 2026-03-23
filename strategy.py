#!/usr/bin/env python3
"""
Experiment #1031: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 748+ failed strategies, the key insight is:
1. Connors RSI (CRSI) has proven edge in bear/range markets (75% win rate in research)
2. Choppiness Index regime filter prevents trend-following in chop (major loss source)
3. 1d HMA21 provides cleaner trend bias than dual-HTF (which failed in #1029)
4. Relaxed CRSI thresholds (15/85 not 10/90) ensure >=30 trades/train
5. Asymmetric logic: easier long entries, harder short (matches 2025 bear bias)

Why this differs from failed attempts:
- SIMPLER than triple-regime strategies (those got 0 trades)
- CRSI instead of Fisher (Fisher failed in #1029 with Sharpe=-0.805)
- Single HTF (1d) not dual (12h+1d failed in #1029)
- Relaxed entry thresholds to guarantee trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Composite of: RSI(3) + RSI(Streak) + PercentRank(100)
    Entry Long: CRSI < 15 (oversold)
    Entry Short: CRSI > 85 (overbought)
    Exit: CRSI crosses 50
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = rsi_3.values
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Component 3: Percent Rank of daily returns over 100 periods
    returns = close_s.pct_change().values
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) == rank_period:
            count_below = np.sum(window[:-1] < returns[i])
            percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h[i]):
            continue
        
        # === MACRO TREND (1d HMA21) ===
        # Asymmetric: long when price > 1d HMA, short when price < 1d HMA
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_4h[i] > 61.8  # Ranging market → mean reversion
        regime_trend = chop_4h[i] < 38.2  # Trending market → trend follow
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 20  # Relaxed from 15 for more trades
        crsi_overbought = crsi_4h[i] > 80  # Relaxed from 85 for more trades
        crsi_cross_long = crsi_4h[i] > 25 and crsi_4h[i-1] <= 20
        crsi_cross_short = crsi_4h[i] < 75 and crsi_4h[i-1] >= 80
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_chop and trend_bull:
            # Mean reversion in choppy market with bullish trend
            if crsi_oversold:
                desired_signal = BASE_SIZE
            elif crsi_cross_long:
                desired_signal = BASE_SIZE
        elif regime_trend and trend_bull:
            # Trend pullback in trending bullish market
            if crsi_4h[i] < 40 and crsi_4h[i-1] >= 40:
                desired_signal = REDUCED_SIZE
        elif not regime_chop and not regime_trend:
            # Neutral regime - only enter on extreme CRSI
            if crsi_oversold and trend_bull:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_chop and trend_bear:
            # Mean reversion in choppy market with bearish trend
            if crsi_overbought:
                desired_signal = -BASE_SIZE
            elif crsi_cross_short:
                desired_signal = -BASE_SIZE
        elif regime_trend and trend_bear:
            # Trend pullback in trending bearish market
            if crsi_4h[i] > 60 and crsi_4h[i-1] <= 60:
                desired_signal = -REDUCED_SIZE
        elif not regime_chop and not regime_trend:
            # Neutral regime - only enter on extreme CRSI
            if crsi_overbought and trend_bear:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long if CRSI goes overbought or trend reverses
        if in_position and position_side > 0:
            if crsi_4h[i] > 70:
                desired_signal = 0.0
            elif not trend_bull and crsi_4h[i] > 50:
                desired_signal = 0.0
        
        # Exit short if CRSI goes oversold or trend reverses
        if in_position and position_side < 0:
            if crsi_4h[i] < 30:
                desired_signal = 0.0
            elif not trend_bear and crsi_4h[i] < 50:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend bullish and CRSI not overbought
                if trend_bull and crsi_4h[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend bearish and CRSI not oversold
                if trend_bear and crsi_4h[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = BASE_SIZE
        elif desired_signal < -0.15:
            desired_signal = -BASE_SIZE
        elif desired_signal > 0:
            desired_signal = REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -REDUCED_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals