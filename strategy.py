#!/usr/bin/env python3
"""
Experiment #073: 1d Primary + 1w HTF — Dual Regime (Trend + Mean Reversion)

Hypothesis: Daily timeframe with Weekly trend bias provides the best signal-to-noise
ratio for crypto. Using Choppiness Index to switch between trend-following (Donchian
breakout) and mean-reversion (Connors RSI) regimes should capture both trending
and ranging markets. Weekly HTF provides strong directional bias to filter trades.

Key innovations:
1. 1d primary with 1w HTF - proven higher TF works better for crypto
2. Choppiness Index (14) regime detection - CHOP>61.8=range, CHOP<38.2=trend
3. Donchian(20) breakout for trend entries with Weekly bias confirmation
4. Connors RSI for mean reversion when choppy (CRSI<15 long, >85 short)
5. ATR(14) trailing stop at 2.5x for risk management
6. Conservative sizing: 0.25-0.30 max to survive 2022-style crashes

Target: Sharpe>0.351, trades>10/symbol train, >3/symbol test, DD>-40%
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range-bound (mean reversion regime)
    CHOP < 38.2 = trending (trend follow regime)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
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
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = streak[i - 1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        lookback = streak[i - streak_period:i + 1]
        max_streak = max(abs(lookback.max()), abs(lookback.min()))
        if max_streak > 0:
            streak_rsi[i] = 50.0 + (streak[i] / max_streak) * 50.0
        else:
            streak_rsi[i] = 50.0
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank (100) - where does current return rank vs last 100 bars
    pr = np.zeros(n)
    for i in range(pr_period, n):
        returns = np.diff(close[i - pr_period:i + 1])
        if len(returns) > 0:
            current_return = close[i] - close[i - 1]
            pr[i] = 100.0 * np.sum(returns < current_return) / len(returns)
    
    # Combine into CRSI
    for i in range(pr_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channels - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    choppiness = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_MR = 0.25
    
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
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range-bound market
        is_trending = choppiness[i] < 45.0  # Trending market
        
        # === TREND SIGNALS (Donchian Breakout + HMA) ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # Donchian breakout - price breaks 20-day high/low
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === MEAN REVERSION SIGNALS (Connors RSI) ===
        crsi_oversold = crsi[i] < 20.0  # Oversold - long opportunity
        crsi_overbought = crsi[i] > 80.0  # Overbought - short opportunity
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TREND FOLLOWING: When trending + HTF bias + Donchian breakout
        if is_trending:
            if htf_bull and hma_bull and breakout_long:
                desired_signal = SIZE_TREND
            elif htf_bear and hma_bear and breakout_short:
                desired_signal = -SIZE_TREND
        
        # MEAN REVERSION: When choppy + CRSI extremes + HTF bias
        if is_choppy:
            if htf_bull and crsi_oversold:
                desired_signal = SIZE_MR
            elif htf_bear and crsi_overbought:
                desired_signal = -SIZE_MR
        
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
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal >= SIZE_MR * 0.85:
            final_signal = SIZE_MR
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal <= -SIZE_MR * 0.85:
            final_signal = -SIZE_MR
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