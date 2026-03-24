#!/usr/bin/env python3
"""
Experiment #786: 1d Primary + 1w HTF — Regime-Adaptive Connors RSI Strategy

Hypothesis: Daily timeframe with weekly trend bias provides optimal signal quality
for crypto perpetuals. Previous 1d strategies failed due to overly complex filters.
This version combines:
1. 1w HMA(21) for long-term trend bias (proven in mtf_hma_rsi_zscore_v1)
2. 1d Choppiness Index for regime detection (trending vs ranging)
3. Connors RSI (CRSI) for entry timing — 75% win rate in literature
4. Regime-adaptive logic: trend-follow in trending, mean-revert in chop
5. 2.5x ATR trailing stop for risk management

CRSI Formula: (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 10 + price > weekly HMA (in bull regime)
- Short: CRSI > 90 + price < weekly HMA (in bear regime)
- Range regime: fade extremes at CRSI < 5 / > 95

Target: Sharpe > 0.40, trades >= 20/train, trades >= 3/test, DD > -40%
Timeframe: 1d
Size: 0.25-0.30 discrete (max 0.35)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
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

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI - measures consecutive up/down days"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like score (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(period, n):
        max_streak = max(abs(streak[max(0, i-period):i+1]).max(), 1)
        current = streak[i]
        streak_rsi[i] = 50.0 + (current / max_streak) * 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI - current return vs past returns"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = returns[max(0, i-period+1):i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        pr[i] = rank * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=2, streak_period=2, pr_period=100):
    """Connors RSI - composite mean reversion indicator"""
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + streak_rsi + pr) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - detects trending vs ranging markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = high[max(0, i-period+1):i+1].max()
        lowest = low[max(0, i-period+1):i+1].min()
        tr_sum = 0.0
        for j in range(max(0, i-period+1), i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        if tr_sum > 1e-10 and (highest - lowest) > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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
    hma_1d = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(250, n):  # Start after 200 SMA + buffer
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        is_ranging = chop[i] > 55.0  # Slightly relaxed threshold
        is_trending = chop[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Mean reversion short
        crsi_extreme_oversold = crsi[i] < 8.0
        crsi_extreme_overbought = crsi[i] > 92.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_trending:
            # Trend-following mode: enter on pullbacks in direction of weekly trend
            if htf_1w_bull and above_sma200:
                if crsi_oversold:
                    if crsi_extreme_oversold:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            elif htf_1w_bear and below_sma200:
                if crsi_overbought:
                    if crsi_extreme_overbought:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        else:
            # Range mode: mean revert at extremes, respect weekly bias
            if htf_1w_bull:
                # Only long in bull range
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                elif crsi_oversold:
                    desired_signal = SIZE_BASE
            elif htf_1w_bear:
                # Only short in bear range
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                elif crsi_overbought:
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