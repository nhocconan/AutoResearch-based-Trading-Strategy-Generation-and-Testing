#!/usr/bin/env python3
"""
Experiment #802: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime

Hypothesis: 4h timeframe with 1d HTF bias provides optimal trade frequency (20-50/year)
with high-quality signals. Connors RSI captures mean reversion in range markets (75% win rate),
while Choppiness Index detects regime to switch between mean revert and trend follow.

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Extreme oversold <10 → long, extreme overbought >90 → short
2. Choppiness Index (CHOP) regime filter
   - CHOP > 61.8 = range (use CRSI mean reversion)
   - CHOP < 38.2 = trend (use HMA trend follow)
3. 1d HMA(21) for HTF trend bias — only trade in HTF direction
4. 4h HMA(16/48) for local trend confirmation
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE CRSI thresholds (15/85 not 10/90) to ensure trade generation

Entry conditions (LOOSE for ≥30 trades/train, ≥3/test):
- LONG: 1d HMA bull + (CRSI<20 in range OR HMA bull crossover in trend)
- SHORT: 1d HMA bear + (CRSI>80 in range OR HMA bear crossover in trend)

Target: Sharpe>0.50, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * (SUM(ATR, period) / (Highest High - Lowest Low)) * 100 / log10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            choppiness[i] = 100.0 * (atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Short-term momentum
    RSI_Streak(2): Consecutive up/down days momentum
    PercentRank(100): Where today's return ranks over last 100 days
    
    CRSI < 10 = extreme oversold (long), CRSI > 90 = extreme overbought (short)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down)
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    streak[0] = 0
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive for RSI calculation
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs, prepend=streak_abs[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_streak[i] = 100.0
    
    # Component 3: PercentRank of returns over last 100 periods
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window_returns <= current_return)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range market (loose threshold for more trades)
        is_trending = choppiness[i] < 45.0  # Trend market (loose threshold)
        
        # === 4h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 4h HMA TREND ===
        hma_4h_bull = hma_16[i] > hma_48[i]
        hma_4h_bear = hma_16[i] < hma_48[i]
        
        # === CRSI CONDITIONS (LOOSE for more trades) ===
        crsi_oversold = crsi[i] < 25.0  # Was 10, now 25 for more trades
        crsi_overbought = crsi[i] > 75.0  # Was 90, now 75 for more trades
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (DUAL REGIME - LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        if htf_1d_bull:
            # LONG in bull regime
            if is_choppy:
                # Mean reversion: buy CRSI oversold
                if crsi_oversold:
                    if crsi_extreme_oversold:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            elif is_trending:
                # Trend follow: buy HMA crossover or HMA bull
                if hma_crossover_long or hma_4h_bull:
                    if hma_crossover_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            else:
                # Neutral regime: use either signal
                if crsi_oversold or hma_crossover_long or hma_4h_bull:
                    if crsi_extreme_oversold or hma_crossover_long:
                        desired_signal = SIZE_STRONG
                    elif crsi_oversold or hma_4h_bull:
                        desired_signal = SIZE_BASE
        
        elif htf_1d_bear:
            # SHORT in bear regime
            if is_choppy:
                # Mean reversion: sell CRSI overbought
                if crsi_overbought:
                    if crsi_extreme_overbought:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            elif is_trending:
                # Trend follow: sell HMA crossover or HMA bear
                if hma_crossover_short or hma_4h_bear:
                    if hma_crossover_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            else:
                # Neutral regime: use either signal
                if crsi_overbought or hma_crossover_short or hma_4h_bear:
                    if crsi_extreme_overbought or hma_crossover_short:
                        desired_signal = -SIZE_STRONG
                    elif crsi_overbought or hma_4h_bear:
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