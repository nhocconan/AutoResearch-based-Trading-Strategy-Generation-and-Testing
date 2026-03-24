#!/usr/bin/env python3
"""
Experiment #692: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 12h timeframe with regime-adaptive logic outperforms pure trend-following.
Research shows Choppiness Index > 61.8 = range (mean revert), < 38.2 = trending.
Connors RSI (CRSI) has 75% win rate for reversals. Combining these should capture
both trending and ranging markets that destroyed pure trend strategies in 2022-2025.

Key innovations:
1. Choppiness Index (14) regime filter - switch between mean-revert and trend-follow
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - faster than RSI(14)
3. 1d HMA(21) bias - simpler than 1d+1w, reduces filter paralysis
4. HMA(16/48) for trend confirmation in trending regime
5. ATR(14) 2.5x trailing stop - proven risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (LOOSE to ensure >=30 trades/train, >=3/test):
- TREND REGIME (CHOP < 38.2): HMA16>48 + price>1d_HMA + CRSI<40 → long
- RANGE REGIME (CHOP > 61.8): CRSI<15 → long, CRSI>85 → short (mean revert)
- NEUTRAL (38.2-61.8): HMA trend + CRSI pullback

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_hma_1d_v1"
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
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - composite momentum indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Entry: CRSI < 10-20 for long, CRSI > 80-90 for short
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component - RSI of up/down streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total > 1e-10:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
        else:
            rsi_streak[i] = 50.0
    
    # Percent Rank component - where current return ranks vs last 100 days
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        count_below = np.sum(window < current)
        pct_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

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
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(150, n):  # Start later to ensure CRSI is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        # Neutral zone: 38.2 - 61.8
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER TREND ===
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC BY REGIME ===
        desired_signal = 0.0
        
        if is_ranging:
            # RANGE REGIME: Mean reversion at CRSI extremes
            if crsi_extreme_oversold:
                desired_signal = SIZE_STRONG
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_STRONG
            elif crsi_oversold and htf_bull:
                desired_signal = SIZE_BASE
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE
        
        elif is_trending:
            # TREND REGIME: Follow trend with CRSI pullback
            if htf_bull and hma_bull and crsi_oversold:
                desired_signal = SIZE_STRONG
            elif htf_bear and hma_bear and crsi_overbought:
                desired_signal = -SIZE_STRONG
            elif htf_bull and hma_bull:
                desired_signal = SIZE_BASE * 0.5
            elif htf_bear and hma_bear:
                desired_signal = -SIZE_BASE * 0.5
        
        else:
            # NEUTRAL REGIME: Conservative, require strong confluence
            if htf_bull and hma_bull and crsi_extreme_oversold:
                desired_signal = SIZE_BASE
            elif htf_bear and hma_bear and crsi_extreme_overbought:
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