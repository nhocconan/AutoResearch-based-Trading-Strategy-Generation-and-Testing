#!/usr/bin/env python3
"""
Experiment #046: 12h Primary + 1d HTF — Dual-Regime HMA + Connors RSI

Hypothesis: After 45 experiments, the winning pattern for 12h timeframe is:
1. 1d HMA for HTF trend bias (proven in #043 Sharpe=0.313)
2. Choppiness Index regime detection (CHOP>55 = range, CHOP<45 = trend)
3. Connors RSI for mean reversion entries in range regime (CRSI<20 long, >80 short)
4. Simple HMA crossover for trend regime (loose thresholds)
5. ATR trailing stop for risk management

Key insight: Dual-regime adapts to market conditions. Range = mean revert, Trend = follow.
This should generate MORE trades than pure trend strategies (which failed with Sharpe=0.000).
Connors RSI has 75% win rate in backtests for mean reversion.

Timeframe: 12h (target 20-50 trades/year = 80-200 trades over 4yr train)
Size: 0.28 (discrete, safe through 77% crash)
Target: Beat Sharpe=0.313, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crsi_chop_regime_1d_v1"
timeframe = "12h"
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
    """RSI - momentum oscillator"""
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    Long when CRSI < 20, Short when CRSI > 80
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
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
        abs_streak = np.abs(streak[i-streak_period+1:i+1])
        if len(abs_streak) > 0 and np.max(abs_streak) > 0:
            # Higher streak = more extreme = higher RSI
            streak_rsi[i] = 100.0 * np.abs(streak[i]) / (np.max(abs_streak) + 1e-10)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - where current return ranks vs last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        returns = np.diff(close[i-pr_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = 100.0 * rank
        else:
            percent_rank[i] = 50.0
    
    # Combine into Connors RSI
    crsi = np.full(n, np.nan)
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
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
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=9)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size - safe through 77% crash
    
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND ===
        hma_12h_bull = close[i] > hma_12h[i]
        hma_12h_bear = close[i] < hma_12h[i]
        hma_fast_above_slow = hma_12h_fast[i] > hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        hma_fast_below_slow = hma_12h_fast[i] < hma_12h[i] if not np.isnan(hma_12h_fast[i]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range regime - mean revert
        is_trending = choppiness[i] < 45.0  # Trend regime - trend follow
        
        # === DESIRED SIGNAL (Dual-Regime Logic) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF + primary trend alignment
        if is_trending:
            # LONG: 1d bull + 12h bull + fast HMA above slow
            if hma_1d_bull and hma_12h_bull and hma_fast_above_slow:
                desired_signal = SIZE
            # SHORT: 1d bear + 12h bear + fast HMA below slow
            elif hma_1d_bear and hma_12h_bear and hma_fast_below_slow:
                desired_signal = -SIZE
        
        # RANGE REGIME: Mean reversion with Connors RSI
        elif is_choppy:
            # LONG: CRSI < 25 (oversold) + 1d HMA not strongly bearish
            if crsi[i] < 25.0 and not hma_1d_bear:
                desired_signal = SIZE
            # SHORT: CRSI > 75 (overbought) + 1d HMA not strongly bullish
            elif crsi[i] > 75.0 and not hma_1d_bull:
                desired_signal = -SIZE
        
        # TRANSITION REGIME: Use RSI filter for entries
        else:
            # LONG: 1d bull + RSI pullback < 50
            if hma_1d_bull and rsi[i] < 50.0:
                desired_signal = SIZE
            # SHORT: 1d bear + RSI rally > 50
            elif hma_1d_bear and rsi[i] > 50.0:
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