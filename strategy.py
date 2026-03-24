#!/usr/bin/env python3
"""
Experiment #152: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 151 failed experiments, the clearest pattern for 12h is:
- Connors RSI (CRSI) has 75% win rate in backtests, excels in bear/range markets (2022, 2025)
- Choppiness Index is the BEST meta-filter for regime detection (ETH Sharpe +0.923 in research)
- Dual-regime approach: mean-revert when choppy, trend-follow when trending
- 1d HMA(50) + 1w HMA(21) provide major trend bias without over-filtering
- Loose CRSI thresholds (15/85 instead of 10/90) ensure sufficient trade generation
- This combines: CRSI mean reversion + CHOP regime + MTF HMA trend bias

Key design choices:
- Timeframe: 12h (25-40 trades/year target, minimal fee drag)
- HTF: 1d HMA(50) + 1w HMA(21) for multi-layer trend bias
- Entry: Connors RSI extremes + Choppiness regime filter + HTF alignment
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Regime: CHOP>55 = range (CRSI mean revert), CHOP<55 = trend (CRSI + breakout)
- Position size: 0.30 (30% of capital, discrete levels)
- Stoploss: 2.5x ATR trailing stop
- Loose filters to ensure >=30 trades on train, >=3 on test, ALL symbols Sharpe>0

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_hma_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak Component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        for j in range(period):
            idx = i - j
            if idx <= 0:
                break
            if close[idx] > close[idx - 1]:
                streak += 1
            elif close[idx] < close[idx - 1]:
                streak -= 1
        
        # Convert streak to 0-100 scale
        # Positive streak = bullish, negative = bearish
        streak_rsi[i] = 50.0 + (streak * 25.0)  # Scale: -2 to +2 maps to 0-100
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component of Connors RSI
    Measures where current price change ranks vs last N periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i-1]
        changes = close[i-period+1:i+1] - close[i-period:i]
        
        # Count how many changes are less than current
        count_less = np.sum(changes[:-1] < current_change)  # Exclude current from comparison
        pct_rank[i] = 100.0 * count_less / (period - 1)
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100, extremes <10 or >90 signal reversals
    """
    rsi_fast = calculate_rsi(close, rsi_period)
    streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_fast + streak + pr) / 3.0
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 55 as threshold for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete levels)
    
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
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === HTF BIAS (1d + 1w HMA) ===
        htf_bull_1d = close[i] > hma_1d_aligned[i]
        htf_bear_1d = close[i] < hma_1d_aligned[i]
        htf_bull_1w = close[i] > hma_1w_aligned[i]
        htf_bear_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias when both agree
        htf_strong_bull = htf_bull_1d and htf_bull_1w
        htf_strong_bear = htf_bear_1d and htf_bear_1w
        htf_neutral = (htf_bull_1d and htf_bear_1w) or (htf_bear_1d and htf_bull_1w)
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # Loose threshold for more trades
        crsi_overbought = crsi[i] > 80.0  # Loose threshold for more trades
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Mean revert on CRSI extremes
            # LONG: CRSI oversold + HTF not strongly bear + HMA not strongly bear
            if crsi_oversold and not htf_strong_bear:
                desired_signal = SIZE
            # SHORT: CRSI overbought + HTF not strongly bull + HMA not strongly bull
            elif crsi_overbought and not htf_strong_bull:
                desired_signal = -SIZE
            # Extreme CRSI entries (override HTF)
            elif crsi_extreme_oversold:
                desired_signal = SIZE * 0.7
            elif crsi_extreme_overbought:
                desired_signal = -SIZE * 0.7
        else:
            # TRENDING REGIME: Follow trend with CRSI pullback entries
            # LONG: HTF bull + CRSI pullback (not overbought) + HMA bull
            if htf_strong_bull and crsi[i] < 60.0 and hma_bull:
                desired_signal = SIZE
            # SHORT: HTF bear + CRSI pullback (not oversold) + HMA bear
            elif htf_strong_bear and crsi[i] > 40.0 and hma_bear:
                desired_signal = -SIZE
            # Breakout confirmation in trend
            elif donchian_breakout_bull and htf_bull_1d and hma_bull:
                desired_signal = SIZE * 0.7
            elif donchian_breakout_bear and htf_bear_1d and hma_bear:
                desired_signal = -SIZE * 0.7
            # Fallback: simple trend following
            elif htf_bull_1d and hma_bull and crsi[i] < 70.0:
                desired_signal = SIZE * 0.5
            elif htf_bear_1d and hma_bear and crsi[i] > 30.0:
                desired_signal = -SIZE * 0.5
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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