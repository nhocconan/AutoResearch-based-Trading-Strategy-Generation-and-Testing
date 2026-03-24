#!/usr/bin/env python3
"""
Experiment #874: 1d Primary + 1w HTF — Dual Regime (Trend/Mean-Revert) + Connors RSI

Hypothesis: Daily timeframe with weekly HTF bias provides optimal balance of signal
quality and trade frequency (20-50 trades/year). Choppiness Index detects regime:
trending markets use HMA crossover entries, ranging markets use Connors RSI mean
reversion. Weekly HMA(21) provides strong directional bias to filter counter-trend
trades. This addresses the 2022 crash and 2025 bear market by adapting to regime.

Key innovations:
1. 1w HMA(21) for HTF trend bias - smooth weekly direction
2. Choppiness Index(14) regime: <45 = trend, >55 = range (with hysteresis)
3. Connors RSI(3,2,100) for mean-reversion entries in range regime
4. HMA(16/48) crossover for trend-follow entries in trend regime
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
7. LOOSE entry conditions to ensure ≥10 trades/train, ≥3/test

Entry conditions (loose for trade frequency):
- TREND REGIME (CHOP<45): LONG = 1w HMA bull + 1d HMA crossover up OR HMA16>HMA48
- TREND REGIME (CHOP<45): SHORT = 1w HMA bear + 1d HMA crossover down OR HMA16<HMA48
- RANGE REGIME (CHOP>55): LONG = 1w HMA bull + CRSI<35
- RANGE REGIME (CHOP>55): SHORT = 1w HMA bear + CRSI>65
- TRANSITION (45<=CHOP<=55): hold previous signal

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentile of today's change vs last 100 days
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_streak[i] = 100.0
    
    # PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        changes = np.diff(close[i - rank_period:i + 1])
        current_change = changes[-1]
        count_below = np.sum(changes[:-1] < current_change)
        percent_rank[i] = count_below / (rank_period - 1) * 100.0
    
    # CRSI
    crsi = np.full(n, np.nan)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 45/55 thresholds with hysteresis
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    
    # Regime hysteresis tracking
    prev_regime = 0  # 0=unknown, 1=trend, 2=range
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION WITH HYSTERESIS ===
        chop_val = chop_14[i]
        if chop_val < 45.0:
            current_regime = 1  # Trend
        elif chop_val > 55.0:
            current_regime = 2  # Range
        else:
            current_regime = prev_regime  # Hold previous
        
        prev_regime = current_regime
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_1d_16[i-1]) and not np.isnan(hma_1d_48[i-1]):
            hma_crossover_long = (hma_1d_16[i-1] <= hma_1d_48[i-1]) and (hma_1d_16[i] > hma_1d_48[i])
            hma_crossover_short = (hma_1d_16[i-1] >= hma_1d_48[i-1]) and (hma_1d_16[i] < hma_1d_48[i])
        
        # === HMA TREND ===
        hma_1d_bull = hma_1d_16[i] > hma_1d_48[i]
        hma_1d_bear = hma_1d_16[i] < hma_1d_48[i]
        
        # === CRSI CONDITIONS (LOOSE for trade frequency) ===
        crsi_oversold = crsi[i] < 35.0  # Mean reversion long
        crsi_overbought = crsi[i] > 65.0  # Mean reversion short
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE) ===
        desired_signal = 0.0
        
        if htf_1w_bull:
            # Bullish HTF bias - prefer longs
            if current_regime == 1:  # Trend regime
                if hma_crossover_long:
                    desired_signal = SIZE_STRONG
                elif hma_1d_bull:
                    desired_signal = SIZE_BASE
            else:  # Range regime
                if crsi_oversold:
                    desired_signal = SIZE_STRONG
                elif crsi[i] < 45.0:
                    desired_signal = SIZE_BASE
        
        elif htf_1w_bear:
            # Bearish HTF bias - prefer shorts
            if current_regime == 1:  # Trend regime
                if hma_crossover_short:
                    desired_signal = -SIZE_STRONG
                elif hma_1d_bear:
                    desired_signal = -SIZE_BASE
            else:  # Range regime
                if crsi_overbought:
                    desired_signal = -SIZE_STRONG
                elif crsi[i] > 55.0:
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