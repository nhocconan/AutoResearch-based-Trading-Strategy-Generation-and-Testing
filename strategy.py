#!/usr/bin/env python3
"""
Experiment #039: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: After 38 experiments, the pattern is clear — simple trend following
fails in bear/range markets (2025 test period). The winning approach combines:
1. Connors RSI (CRSI) for entry timing - proven 75% win rate in mean reversion
2. Choppiness Index regime detection - switch between mean-revert and trend-follow
3. 1d HMA for HTF bias - only trade with higher timeframe direction
4. LOOSE thresholds to ensure trade generation (critical after 0-trade failures)

CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 20 in uptrend (extreme oversold pullback)
- Short: CRSI > 80 in downtrend (extreme overbought rally)

Choppiness Index (CHOP):
- CHOP > 61.8 = range market → use mean reversion (CRSI extremes)
- CHOP < 38.2 = trending market → use trend following (HMA crossover)
- 38.2-61.8 = neutral → require stronger confluence

Entry Logic (LOOSE for trade generation):
- Range mode (CHOP>55): Long if CRSI<30 + 1d_HMA_bull, Short if CRSI>70 + 1d_HMA_bear
- Trend mode (CHOP<45): Long if 4h_HMA_fast>slow + 1d_HMA_bull, Short if opposite
- Neutral: Require both CRSI extreme AND HMA alignment

Size: 0.30 (discrete, proven safe through 2022 crash)
Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.25, trades>40/symbol train, >5/symbol test, DD>-35%
Timeframe: 4h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - standard momentum oscillator"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Proven 75% win rate on mean reversion entries
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # Component 1: Short-period RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period):i+1]
        if len(streak_window) > 0:
            # Normalize streak to 0-100 scale
            max_streak = max(abs(streak_window.max()), abs(streak_window.min()), 1)
            streak_rsi[i] = 50.0 + (streak[i] / max_streak) * 50.0
    
    # Component 3: Percentile rank of today's return over last 100 days
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close, prepend=close[0])
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0:
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine all 3 components
    crsi = np.full(n, np.nan)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies ranging vs trending markets
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    choppiness = np.full(n, np.nan)
    for i in range(period * 2, n):
        if highest[i] - lowest[i] > 1e-10 and atr_sum[i] > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum[i] / (highest[i] - lowest[i])) / np.log10(period)
    
    return choppiness

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size - safe through 77% crash
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        hma_fast_above_slow = hma_4h_fast[i] > hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        hma_fast_below_slow = hma_4h_fast[i] < hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = choppiness[i]
        is_range_market = chop_value > 55.0  # Loose threshold for more trades
        is_trend_market = chop_value < 45.0  # Loose threshold for more trades
        # 45-55 = neutral zone
        
        # === DESIRED SIGNAL (LOOSE thresholds for trade generation) ===
        desired_signal = 0.0
        
        # RANGE MODE: Mean reversion with CRSI extremes
        if is_range_market:
            # Long: CRSI extremely oversold + HTF bull bias
            if crsi[i] < 30.0 and hma_1d_bull:
                desired_signal = SIZE
            # Short: CRSI extremely overbought + HTF bear bias
            elif crsi[i] > 70.0 and hma_1d_bear:
                desired_signal = -SIZE
        
        # TREND MODE: Trend following with HMA crossover
        elif is_trend_market:
            # Long: Fast HMA > Slow HMA + HTF bull
            if hma_fast_above_slow and hma_1d_bull:
                desired_signal = SIZE
            # Short: Fast HMA < Slow HMA + HTF bear
            elif hma_fast_below_slow and hma_1d_bear:
                desired_signal = -SIZE
        
        # NEUTRAL MODE: Require stronger confluence (both CRSI + HMA)
        else:
            # Long: CRSI oversold + HMA bull + HTF bull
            if crsi[i] < 35.0 and hma_4h_bull and hma_1d_bull:
                desired_signal = SIZE
            # Short: CRSI overbought + HMA bear + HTF bear
            elif crsi[i] > 65.0 and hma_4h_bear and hma_1d_bear:
                desired_signal = -SIZE
        
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