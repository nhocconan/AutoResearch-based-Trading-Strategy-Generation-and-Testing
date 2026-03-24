#!/usr/bin/env python3
"""
Experiment #120: 6h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + Weekly Trend

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). Key insight from
110+ failed experiments: pure trend fails in bear markets, pure mean-revert fails on SOL.
SOLUTION: Triple-layer regime system:
- 1w HMA: Major trend bias (bull/bear market identification)
- 1d Choppiness: Regime detection (trending vs ranging)
- 6h Connors RSI (CRSI): Entry timing with 75% win rate in research

CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 15 (oversold) + weekly bull bias
- Short when CRSI > 85 (overbought) + weekly bear bias
- In choppy regime: mean revert at CRSI extremes
- In trending regime: follow weekly direction on CRSI pullbacks

Key design choices:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1w HMA for major trend, 1d CHOP for regime
- Entry: CRSI extremes + regime filter + volume confirmation
- Position size: 0.28 (28% of capital, conservative)
- Stoploss: 2.5x ATR trailing
- LOOSE CRSI thresholds (15/85) to ensure trades generate on all symbols

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, >=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_hma_1d1w_v1"
timeframe = "6h"
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
    RSI Streak Component for Connors RSI
    Measures consecutive up/down days
    Score: +period for consecutive up, -period for consecutive down, 0 for flat
    Normalized to 0-100 scale
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        for j in range(period):
            if close[i-j] > close[i-j-1]:
                streak += 1
            elif close[i-j] < close[i-j-1]:
                streak -= 1
        
        # Normalize: period consecutive up = 100, period consecutive down = 0
        streak_rsi[i] = 50.0 + (streak / period) * 50.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0.0, 100.0)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component for Connors RSI
    Measures where current return ranks vs last period returns
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = (count_below / (period - 1)) * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100, <10 = oversold, >90 = overbought
    """
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if np.isnan(rsi_short[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate primary (6h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=50)
    atr = calculate_atr(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness) ===
        # CHOP > 55 = range/choppy (mean revert), CHOP < 55 = trending (trend follow)
        is_choppy = chop_1d_aligned[i] > 55.0
        is_trending = chop_1d_aligned[i] <= 55.0
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # LOOSE threshold for trades
        crsi_overbought = crsi[i] > 80.0  # LOOSE threshold for trades
        crsi_extreme_oversold = crsi[i] < 12.0
        crsi_extreme_overbought = crsi[i] > 88.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === DESIRED SIGNAL (Triple-Layer Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow weekly direction on CRSI pullbacks
            # LONG: weekly bull + CRSI pullback (not extreme) + HMA bull + volume
            if htf_bull and crsi[i] < 45.0 and hma_bull and vol_confirmed:
                desired_signal = SIZE
            # SHORT: weekly bear + CRSI pullback (not extreme) + HMA bear + volume
            elif htf_bear and crsi[i] > 55.0 and hma_bear and vol_confirmed:
                desired_signal = -SIZE
            # Strong breakout: extreme CRSI in trend direction
            elif htf_bull and crsi_extreme_oversold and vol_confirmed:
                desired_signal = SIZE
            elif htf_bear and crsi_extreme_overbought and vol_confirmed:
                desired_signal = -SIZE
        else:
            # CHOPPY REGIME: Mean revert at CRSI extremes
            # LONG: CRSI oversold + not strongly bear weekly
            if crsi_oversold and not htf_bear:
                desired_signal = SIZE
            # SHORT: CRSI overbought + not strongly bull weekly
            elif crsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # Extreme mean reversion (ignore HTF)
            elif crsi_extreme_oversold:
                desired_signal = SIZE * 0.8
            elif crsi_extreme_overbought:
                desired_signal = -SIZE * 0.8
        
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