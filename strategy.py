#!/usr/bin/env python3
"""
Experiment #1600: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI Strategy

Hypothesis: After 11 failed 4h experiments and multiple 1h/30m failures with 0 trades,
the key is balancing regime detection with tradeable entry conditions.

CRITICAL LESSON: Most recent 1h/30m strategies got Sharpe=0.000 (ZERO trades).
Entry conditions were TOO STRICT. This strategy LOOSENS thresholds while keeping
HTF trend filter for direction.

This strategy combines:
1. 4h HMA(21) for trend direction (proven in best strategies)
2. 12h HMA(21) for higher-timeframe bias
3. Connors RSI (CRSI) for entry timing - combines RSI(3) + RSI_Streak(2) + PercentRank(100)
4. Choppiness Index(14) for regime detection - CHOP>55=range, CHOP<45=trend
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete position sizing (0.25) for 1h timeframe

Key innovation: ADAPTIVE entry logic based on regime
- Range (CHOP>55): Mean revert with CRSI<20 long, CRSI>80 short (LOOSENED from 10/90)
- Trend (CHOP<45): Pullback entries with CRSI<40 long (in uptrend), CRSI>60 short (in downtrend)

Why this should work:
- CRSI has 75% win rate in academic studies for mean reversion
- Choppiness Index properly distinguishes regime (unlike simple ADX)
- 4h/12h HMA filter prevents counter-trend trades
- 1h timeframe with 0.25 size = ~40-60 trades/year target
- LOOSENED thresholds ensure we get trades (learned from 0-trade failures)

Timeframe: 1h (required for this experiment)
HTF: 4h HMA + 12h HMA for bias (use mtf_data helper)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_4h12h_hma_chop_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signal
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Fast momentum
    RSI_Streak(2): Consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 bars
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_close[loss_smooth <= 1e-10] = 100.0
    rsi_close[:rsi_period] = np.nan
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask] / streak_loss_smooth[mask]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    rsi_streak[:streak_period+1] = np.nan
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine all three
    crsi = np.full(n, np.nan)
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and tr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
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
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for higher-timeframe bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (4h HMA + 12h HMA) ===
        hma4h_bull = close[i] > hma_4h_aligned[i]
        hma4h_bear = close[i] < hma_4h_aligned[i]
        hma12h_bull = close[i] > hma_12h_aligned[i]
        hma12h_bear = close[i] < hma_12h_aligned[i]
        sma200_bull = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Choppy/ranging market
        is_trend = chop[i] < 45.0  # Trending market
        # Neutral zone 45-55: use trend-following logic
        
        # === CRSI ENTRY SIGNALS (LOOSENED for trade generation) ===
        crsi_oversold = crsi[i] < 20.0  # Mean reversion long
        crsi_overbought = crsi[i] > 80.0  # Mean reversion short
        crsi_pullback_long = crsi[i] < 40.0  # Trend pullback long
        crsi_pullback_short = crsi[i] > 60.0  # Trend pullback short
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if hma4h_bull and hma12h_bull and sma200_bull:
            if is_range and crsi_oversold:
                # Range market: mean reversion from oversold
                desired_signal = BASE_SIZE
            elif (is_trend or not is_range) and crsi_pullback_long:
                # Trending market: buy pullback in uptrend
                desired_signal = BASE_SIZE
            elif not is_range and not is_trend and crsi_pullback_long:
                # Neutral zone: follow trend with pullback
                desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        elif hma4h_bear and hma12h_bear:
            if is_range and crsi_overbought:
                # Range market: mean reversion from overbought
                desired_signal = -BASE_SIZE
            elif (is_trend or not is_range) and crsi_pullback_short:
                # Trending market: sell pullback in downtrend
                desired_signal = -BASE_SIZE
            elif not is_range and not is_trend and crsi_pullback_short:
                # Neutral zone: follow trend with pullback
                desired_signal = -BASE_SIZE
        
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
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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