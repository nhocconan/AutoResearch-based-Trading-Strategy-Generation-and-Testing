#!/usr/bin/env python3
"""
Experiment #552: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 12h timeframe provides optimal trade frequency (20-50/year) with reduced
fee drag. Choppiness Index cleanly separates trend vs range regimes. Connors RSI
(CRSI) provides superior mean-reversion signals vs standard RSI (75% win rate in
literature). Combined with 1d HMA for trend bias, this should work in both bull
and bear markets.

Key improvements over failed #544 (mtf_12h_regime_hma_donchian_rsi_1d1w_v1):
1. Connors RSI instead of standard RSI - better mean reversion signals
2. Simpler entry logic - fewer filters = more trades (avoid 0-trade failure)
3. Clearer regime thresholds (CHOP>61.8 range, CHOP<38.2 trend)
4. Asymmetric sizing - stronger signals in confirmed trends
5. Looser RSI thresholds to ensure trade generation

Strategy logic:
1. 1d HMA(21) = trend bias (price above = bullish bias)
2. 12h HMA(16) = fast trend following
3. 12h Choppiness(14) = regime detection
4. 12h Connors RSI = entry timing (CRSI<10 long, CRSI>90 short)
5. ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- TREND (CHOP<38.2): Follow HMA direction with HTF confirmation
- RANGE (CHOP>61.8): Mean revert at CRSI extremes
- TRANSITION (38-62): Reduced size, require stronger signals

Target: Sharpe>0.40, trades>=30 train (7.5/year), trades>=3 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d_v1"
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

def calculate_streak_rsi(close, period=2):
    """
    Connors RSI Streak Component
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        for j in range(1, period + 1):
            if close[i - j + 1] > close[i - j]:
                streak += 1
            elif close[i - j + 1] < close[i - j]:
                streak -= 1
        
        # Convert streak to RSI-like value (0-100)
        # Positive streak = higher values, negative = lower
        streak_rsi[i] = 50.0 + streak * 25.0
        streak_rsi[i] = max(0.0, min(100.0, streak_rsi[i]))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank Component
    Current close percentile vs last N periods
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i - period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long signal)
    CRSI > 90 = overbought (short signal)
    """
    rsi_fast = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_streak_rsi(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
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
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=16)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        
        # === HTF BIAS (1d trend) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # HMA slope
        hma_slope_bull = hma_12h[i] > hma_12h[i-3] if i >= 3 and not np.isnan(hma_12h[i-3]) else False
        hma_slope_bear = hma_12h[i] < hma_12h[i-3] if i >= 3 and not np.isnan(hma_12h[i-3]) else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 61.8    # Range-bound (mean reversion)
        chop_trend = chop[i] < 38.2    # Trending (trend follow)
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === REGIME DETECTION ===
        is_trend_regime = chop_trend
        is_range_regime = chop_range
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HMA direction with HTF confirmation
        if is_trend_regime:
            # Strong long: HTF bull + HMA bull + HMA rising
            if htf_bull and hma_bull and hma_slope_bull:
                desired_signal = SIZE_STRONG
            # Strong short: HTF bear + HMA bear + HMA falling
            elif htf_bear and hma_bear and hma_slope_bear:
                desired_signal = -SIZE_STRONG
            # Moderate long: HTF bull + HMA bull (no slope requirement)
            elif htf_bull and hma_bull:
                desired_signal = SIZE_BASE
            # Moderate short: HTF bear + HMA bear
            elif htf_bear and hma_bear:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at CRSI extremes
        elif is_range_regime:
            # Long at extreme oversold
            if crsi_extreme_oversold:
                desired_signal = SIZE_BASE
            # Short at extreme overbought
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_BASE
            # Long at oversold with HTF support
            elif crsi_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            # Short at overbought with HTF resistance
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # TRANSITION REGIME: Require stronger signals
        else:
            # Only enter on extreme CRSI + HTF alignment
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.7
            elif crsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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