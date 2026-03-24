#!/usr/bin/env python3
"""
Experiment #1553: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Based on experiment history, 1d timeframe strategies work best (#1543 Sharpe=0.618).
Lower TFs (4h, 12h, 30m, 1h) consistently fail with negative Sharpe or 0 trades.
Research shows Connors RSI (CRSI) has 75% win rate for mean reversion entries.

Strategy Design:
- HTF Bias: 1w HMA(21) for macro trend direction (proven in #1543)
- Primary: 1d Connors RSI for mean reversion entry timing
- Long: CRSI < 15 (oversold) + price > 1w HMA (uptrend pullback)
- Short: CRSI > 85 (overbought) + price < 1w HMA (downtrend rally)
- Exit: CRSI crosses 50 (mean reversion complete) OR 2.5x ATR stoploss
- Size: 0.28 discrete (0.0, ±0.28) to minimize fee churn
- Target: 20-50 trades/year, Sharpe > 0.618

Why this should beat #1543:
- CRSI is more sensitive than RSI(14) for catching reversals
- Mean reversion works better in 2025 bear/range market than pure trend
- 1w HMA filter prevents counter-trend trades (major improvement over no HTF)
- Loose CRSI thresholds (15/85 not 10/90) ensure trades fire

Timeframe: 1d (proven best TF from experiment history)
HTF: 1w HMA(21) for bias only
Target: Sharpe > 0.618, trades > 30/train, > 3/test, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_meanrev_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    delta = np.diff(streak, prepend=streak[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    streak_rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    streak_rsi[loss_smooth <= 1e-10] = 100.0
    streak_rsi[:period] = np.nan
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures where current return ranks vs last N days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan)
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        current = window[-1]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    More sensitive than standard RSI for mean reversion
    """
    n = len(close)
    
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pr = calculate_percent_rank(close, period=pr_period)
    
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(pr)
    crsi[mask] = (rsi_3[mask] + streak_rsi[mask] + pr[mask]) / 3.0
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss and exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI percent rank
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Long entry (loose threshold for trades)
        crsi_overbought = crsi[i] > 85.0  # Short entry
        crsi_neutral = 40.0 < crsi[i] < 60.0  # Exit zone
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + CRSI oversold (uptrend pullback)
        if weekly_bull and crsi_oversold:
            desired_signal = BASE_SIZE
        
        # SHORT: Weekly bear + CRSI overbought (downtrend rally)
        if weekly_bear and crsi_overbought:
            desired_signal = -BASE_SIZE
        
        # === EXIT LOGIC (CRSI mean reversion complete) ===
        if in_position:
            if position_side > 0 and crsi_neutral:
                desired_signal = 0.0  # Long exit
            if position_side < 0 and crsi_neutral:
                desired_signal = 0.0  # Short exit
        
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