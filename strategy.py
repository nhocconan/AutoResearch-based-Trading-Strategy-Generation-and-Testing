#!/usr/bin/env python3
"""
Experiment #1542: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + Connors RSI + Donchian

Hypothesis: After 11 failed 4h experiments and reviewing 12h history (#1532 Sharpe=0.119, #1536 Sharpe=0.009),
the pattern shows 12h needs SIMPLER logic with regime-aware entries:

1. Choppiness Index (CHOP) detects regime: >61.8 = range (mean revert), <38.2 = trend (breakout)
2. Connors RSI with LOOSE thresholds (<35 long, >65 short) — NOT <20/>80 which caused 0 trades
3. 1d HMA(21) for macro bias — single HTF filter avoids conflicts
4. Donchian(20) breakout in trending regime, Bollinger mean-revert in choppy regime
5. 1w HMA for ultra-long-term bias filter (only trade with weekly trend)

Why this should work on 12h:
- 12h = 2 bars/day = ~730 bars/year = natural fit for 20-50 trades/year target
- Dual regime adapts to market state (ETH research: CHOP + CRSI Sharpe +0.923)
- Loose CRSI thresholds ensure trades fire (≥10 train, ≥3 test per symbol)
- 1d + 1w HMA confluence prevents counter-trend trades in strong macro moves
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn at 12h frequency

Timeframe: 12h (required by experiment #1542)
HTF: 1d HMA(21) + 1w HMA(21) for macro bias
Position Size: 0.28 (conservative for 12h volatility)
Target: Sharpe > 0.618, DD < -40%, trades > 30/train, > 3/test per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (breakout)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme < 10 = oversold (long), > 90 = overbought (short)
    We use LOOSE thresholds: < 35 long, > 65 short to ensure trades fire
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_3 = np.full(n, np.nan)
    mask = loss_3 > 1e-10
    rsi_3[mask] = 100.0 - (100.0 / (1.0 + gain_3[mask] / loss_3[mask]))
    rsi_3[loss_3 <= 1e-10] = 100.0
    rsi_3[:rsi_period] = np.nan
    
    # RSI Streak (2) — streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_2 > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_2[mask] / streak_loss_2[mask]))
    rsi_streak[streak_loss_2 <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # Percent Rank (100) — where current return ranks vs last 100
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel breakout levels"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion"""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    bb_middle, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range-bound market (mean revert)
        is_trending = chop[i] < 45.0  # Trending market (breakout)
        # Neutral zone 45-55 = no trades or reduced size
        
        # === MACRO TREND BIAS (1d + 1w HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bias only when 1d and 1w agree
        strong_bull = daily_bull and weekly_bull
        strong_bear = daily_bear and weekly_bear
        
        # === CONNORS RSI (LOOSE thresholds for trade frequency) ===
        crsi_oversold = crsi[i] < 35.0  # Long signal (loose vs <10)
        crsi_overbought = crsi[i] > 65.0  # Short signal (loose vs >90)
        crsi_extreme_long = crsi[i] < 20.0  # Strong long
        crsi_extreme_short = crsi[i] > 80.0  # Strong short
        
        # === DONCHIAN BREAKOUT (for trending regime) ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === BOLLINGER MEAN REVERSION (for choppy regime) ===
        bb_revert_long = close[i] < bb_lower[i] if not np.isnan(bb_lower[i]) else False
        bb_revert_short = close[i] > bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === DESIRED SIGNAL — REGIME-AWARE ENTRY ===
        desired_signal = 0.0
        signal_strength = 0
        
        if is_trending:
            # TRENDING REGIME: Use Donchian breakout + macro bias
            if breakout_long and strong_bull:
                signal_strength = 3
                if crsi_oversold:
                    signal_strength += 1
                if crsi_extreme_long:
                    signal_strength += 1
            elif breakout_short and strong_bear:
                signal_strength = -3
                if crsi_overbought:
                    signal_strength -= 1
                if crsi_extreme_short:
                    signal_strength -= 1
            # Weaker signals without full confluence
            elif breakout_long and daily_bull:
                signal_strength = 2
            elif breakout_short and daily_bear:
                signal_strength = -2
        
        elif is_choppy:
            # CHOPPY REGIME: Use Bollinger mean reversion + CRSI
            if bb_revert_long and crsi_oversold:
                signal_strength = 3
                if crsi_extreme_long:
                    signal_strength += 1
            elif bb_revert_short and crsi_overbought:
                signal_strength = -3
                if crsi_extreme_short:
                    signal_strength -= 1
            # Weaker signals
            elif bb_revert_long:
                signal_strength = 2
            elif bb_revert_short:
                signal_strength = -2
        
        # Convert strength to signal
        if signal_strength >= 4:
            desired_signal = BASE_SIZE
        elif signal_strength >= 3:
            desired_signal = BASE_SIZE
        elif signal_strength >= 2:
            desired_signal = BASE_SIZE * 0.7
        elif signal_strength <= -4:
            desired_signal = -BASE_SIZE
        elif signal_strength <= -3:
            desired_signal = -BASE_SIZE
        elif signal_strength <= -2:
            desired_signal = -BASE_SIZE * 0.7
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.35:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.35:
            final_signal = -BASE_SIZE * 0.5
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