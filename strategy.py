#!/usr/bin/env python3
"""
Experiment #464: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI

Hypothesis: 12h timeframe reduces noise vs 6h while maintaining trade frequency.
Recent 6h failures show dual-HTF filters are too restrictive (0 trades).
This strategy uses:
1. CHOPPINESS INDEX (14) for regime: >61.8 = range (mean revert), <38.2 = trend
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — 75% win rate
3. 1d HMA(21) for trend bias (single HTF, not dual — looser)
4. 1w HMA(21) for macro bias confirmation
5. LOOSE entries: CRSI<15 long, CRSI>85 short (wider than typical 10/90)
6. ATR(14) stoploss at 2.5x for capital preservation

Why this should work:
- Choppiness filter prevents trend strategies in range markets (2022-2024)
- Connors RSI proven for mean reversion in bear/range markets
- 12h TF = ~30-50 trades/year target (fee-efficient)
- Single HTF bias (1d) instead of dual = more trades qualify
- 1w HMA only for macro filter (not strict requirement)

Target: Sharpe>0.45, DD>-35%, trades>=80 train (20/year), trades>=12 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_connors_hma_1d1w_v1"
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
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar (true range)
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
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_streak_rsi(close, period=2):
    """
    Connors RSI Component 2: Streak RSI
    Measures duration of current up/down streak
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak[0] = 1
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (absolute streak length)
    abs_streak = np.abs(streak)
    
    # Calculate RSI of streak values
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        window = abs_streak[i-period+1:i+1]
        if len(window) > 0:
            rank = np.sum(window < abs_streak[i]) / len(window)
            streak_rsi[i] = rank * 100.0
    
    return streak_rsi

def calculate_percentile_rank(values, period=100):
    """
    Connors RSI Component 3: Percentile Rank of close over lookback period
    """
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid) * 100.0
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with ~75% win rate
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_close = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_streak_rsi(close, streak_period)
    pr = calculate_percentile_rank(close, pr_period)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(350, n):
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
        
        if np.isnan(sma_200[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range (mean reversion mode)
        # CHOP < 38.2 = trending (trend follow mode)
        # Between = maintain previous regime
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === MACRO BIAS (1w HMA) — soft filter only ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (LOOSE: 15/85 instead of 10/90) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === 12h HMA DIRECTION ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with CRSI)
        if is_choppy:
            # Long: CRSI < 15 + above SMA200 (avoid catching falling knife)
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            # Short: CRSI > 85 + below SMA200
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            # Extra: Extreme CRSI without SMA filter (more trades)
            elif crsi[i] < 10.0:
                desired_signal = SIZE_BASE
            elif crsi[i] > 90.0:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (follow HTF bias with CRSI pullback)
        elif is_trending:
            # Long: 1d HMA bull + CRSI pullback < 40 + 12h HMA bull
            if htf_1d_bull and crsi[i] < 40.0 and hma_bull:
                desired_signal = SIZE_STRONG
            # Short: 1d HMA bear + CRSI rally > 60 + 12h HMA bear
            elif htf_1d_bear and crsi[i] > 60.0 and hma_bear:
                desired_signal = -SIZE_STRONG
        
        # REGIME 3: TRANSITION (between 38.2-61.8) — use HTF bias only
        else:
            # Long: 1d bull + 1w bull + CRSI < 50
            if htf_1d_bull and htf_1w_bull and crsi[i] < 50.0:
                desired_signal = SIZE_BASE
            # Short: 1d bear + 1w bear + CRSI > 50
            elif htf_1d_bear and htf_1w_bear and crsi[i] > 50.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals