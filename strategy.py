#!/usr/bin/env python3
"""
Experiment #712: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 12h timeframe with regime-adaptive logic can capture both trending and ranging markets.
Choppiness Index detects regime (CHOP>61.8=range, CHOP<38.2=trend), Connors RSI provides
high-probability mean-reversion entries in ranges, HMA crossover confirms trend direction.
1d HMA provides HTF bias filter to avoid counter-trend trades.

Key innovations:
1. Choppiness Index(14) regime detection - switches between mean-revert and trend-follow
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven 75% win rate
3. 1d HMA(21) HTF bias - only trade with higher timeframe direction
4. 12h HMA(21/63) primary trend - medium-term trend confirmation
5. ATR(14) 2.5x trailing stop - risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
7. LOOSE entry thresholds to ensure >=30 trades/year

Entry conditions (LOOSE to ensure trade generation):
- RANGE (CHOP>55): Long CRSI<25, Short CRSI>75 (mean reversion)
- TREND (CHOP<45): Long if 1d HMA bull + 12h HMA bull + CRSI<50
                     Short if 1d HMA bear + 12h HMA bear + CRSI>50
- TRANSITION (45-55): No new entries, hold existing positions

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d_v1"
timeframe = "12h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - combines RSI, streak RSI, and percent rank"""
    n = len(close)
    if n < rank_period + rsi_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * abs_streak / (abs_streak + 1)
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * 1 / (abs_streak + 1)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank(100) - where does today's return rank vs last 100 days
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[max(0, i-rank_period):i+1])
        if len(returns) > 0:
            today_return = close[i] - close[i-1] if i > 0 else 0
            count_below = np.sum(returns < today_return)
            percent_rank[i] = 100.0 * count_below / len(returns)
    
    # Connors RSI = average of three components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending (0-100)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                if j > 0:
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                    atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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
    
    # Calculate 12h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_63 = calculate_hma(close, period=63)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range, CHOP < 38.2 = trend, between = transition
        # Using looser thresholds (55/45) to get more regime signals
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        is_transition = not is_range and not is_trend
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bull = hma_21[i] > hma_63[i]
        hma_bear = hma_21[i] < hma_63[i]
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 20 = oversold, CRSI > 80 = overbought
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_moderate_low = crsi[i] < 50.0
        crsi_moderate_high = crsi[i] > 50.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # RANGE REGIME - Mean Reversion (LOOSE thresholds)
        if is_range:
            if crsi_oversold and htf_1d_bull:
                desired_signal = SIZE_STRONG
            elif crsi_oversold:
                desired_signal = SIZE_BASE
            elif crsi_overbought and htf_1d_bear:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME - Trend Following with CRSI pullback entry
        elif is_trend:
            # Long: HTF bull + HMA bull + CRSI pullback (not extreme)
            if htf_1d_bull and hma_bull and crsi_moderate_low:
                desired_signal = SIZE_STRONG
            elif htf_1d_bull and hma_bull:
                desired_signal = SIZE_BASE
            
            # Short: HTF bear + HMA bear + CRSI rally (not extreme)
            elif htf_1d_bear and hma_bear and crsi_moderate_high:
                desired_signal = -SIZE_STRONG
            elif htf_1d_bear and hma_bear:
                desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME - Hold existing, no new entries
        elif is_transition:
            if in_position:
                desired_signal = signals[i-1] if i > 0 else 0.0
            else:
                desired_signal = 0.0
        
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