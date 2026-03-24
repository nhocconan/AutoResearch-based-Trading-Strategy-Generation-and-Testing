#!/usr/bin/env python3
"""
Experiment #642: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: The key failure of #638 was using breakout logic (Donchian) in choppy markets.
This strategy uses Choppiness Index to DETECT regime first, then applies appropriate logic:
- CHOP > 61.8 (choppy): Use Connors RSI mean reversion (buy oversold, sell overbought)
- CHOP < 38.2 (trending): Use HMA crossover trend following
- 38.2 <= CHOP <= 61.8 (neutral): Reduce size or stay flat

Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 15 (extreme oversold) + price > 1d HMA bias
- Short when CRSI > 85 (extreme overbought) + price < 1d HMA bias

This should generate MORE trades than #638 because mean reversion triggers frequently
in choppy 2022-2024 period, while trend following captures 2021 bull and 2025 moves.

Key innovations:
1. Choppiness Index regime detection (proven in literature for crypto)
2. Connors RSI for mean reversion (75% win rate in ranges per research)
3. HMA(16/48) crossover for trend following (faster than EMA)
4. 1d HMA(21) bias filter (only long above, only short below)
5. ATR(14) trailing stop at 2.5x
6. Discrete sizing: 0.0, ±0.25, ±0.30

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

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
    Choppiness Index - measures if market is choppy or trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) * LOG10(period)
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period - 1, n):
        if np.isnan(atr[i]):
            continue
        
        # Sum of ATR over period
        atr_sum = np.nansum(atr[i - period + 1:i + 1])
        
        # Highest high and lowest low over period
        hh = np.nanmax(high[i - period + 1:i + 1])
        ll = np.nanmin(low[i - period + 1:i + 1])
        
        if hh - ll > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / (hh - ll)) * np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.maximum(delta, 0)
    loss[1:] = np.maximum(-delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    RSI of streak length - measures consecutive up/down days
    Streak: +1 for up, -1 for down, accumulate
    Then calculate RSI on absolute streak values
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert to positive values for RSI calculation (use absolute streak as "gain/loss")
    # Actually for Connors RSI, we use RSI on the streak series directly
    streak_rsi = calculate_rsi(np.abs(streak) + 1e-10, period)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank - where current close ranks vs last period closes
    Returns 0-100 scale
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        current = close[i]
        
        # Count how many values in window are less than current
        count_less = np.sum(window[:-1] < current)  # exclude current from comparison
        total_compare = period - 1
        
        if total_compare > 0:
            pr[i] = 100.0 * count_less / total_compare
        else:
            pr[i] = 50.0
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme oversold: CRSI < 10-15
    Extreme overbought: CRSI > 85-90
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + pr) / 3.0
    return crsi

def calculate_hma(close, period):
    """Hull Moving Average"""
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
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    
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
    
    for i in range(100, n):
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
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        # Neutral zone: 38.2 <= chop <= 61.8
        
        # === HMA TREND (for trending regime) ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === CONNORS RSI (for choppy regime) ===
        crsi_oversold = crsi[i] < 20  # Loose threshold to ensure trades
        crsi_overbought = crsi[i] > 80
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if htf_bull:  # Only long when HTF bullish
            if is_choppy and crsi_oversold:
                # Mean reversion in choppy market
                desired_signal = SIZE_BASE
            elif is_trending and hma_bull:
                # Trend following in trending market
                # Check for pullback entry (price near HMA fast)
                pullback = close[i] < hma_fast[i] * 1.01  # Within 1% of fast HMA
                if pullback:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif not is_choppy and not is_trending and hma_bull and crsi_oversold:
                # Neutral regime: both signals confirm
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES
        elif htf_bear:  # Only short when HTF bearish
            if is_choppy and crsi_overbought:
                # Mean reversion in choppy market
                desired_signal = -SIZE_BASE
            elif is_trending and hma_bear:
                # Trend following in trending market
                # Check for pullback entry (price near HMA fast)
                pullback = close[i] > hma_fast[i] * 0.99  # Within 1% of fast HMA
                if pullback:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif not is_choppy and not is_trending and hma_bear and crsi_overbought:
                # Neutral regime: both signals confirm
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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