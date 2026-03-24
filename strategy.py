#!/usr/bin/env python3
"""
Experiment #446: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with weekly HTF bias should capture multi-week trends
while avoiding the noise of lower TFs. Recent 1d failures show:
- Too many filters = 0 trades (experiments #434, #436, #439, #441, #445)
- Weekly pivot filters too restrictive
- Need LOOSE entry conditions to generate trades on daily bars

New approach based on proven 1d patterns:
1. CONNORS RSI (CRSI): 3-component mean reversion signal (RSI3 + StreakRSI + PercentRank)
   Entry: CRSI < 15 (long) or CRSI > 85 (short) — proven 75% win rate
2. CHOPPINESS INDEX regime: CHOP > 61.8 = range (use CRSI), CHOP < 38.2 = trend (use HMA)
3. HMA trend filter: 1w HMA for primary bias, only trade in direction of weekly trend
4. DONCHIAN breakout: 20-day breakout for trend entries when CHOP < 38.2
5. LOOSE conditions: Max 2-3 filters to ensure trades generate

Entry Logic:
- Range Long: CHOP > 61.8 + CRSI < 15 + price > SMA200
- Range Short: CHOP > 61.8 + CRSI > 85 + price < SMA200
- Trend Long: CHOP < 38.2 + 1w HMA bull + Donchian breakout OR HMA cross
- Trend Short: CHOP < 38.2 + 1w HMA bear + Donchian breakdown OR HMA cross

Target: Sharpe>0.45, DD>-35%, trades>=40 train (10/year), trades>=6 test
Timeframe: 1d (as specified for experiment #446)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_donchian_1w_v1"
timeframe = "1d"
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
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - 3-component mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive days of gains (positive) or losses (negative)
    PercentRank: percentile rank of today's return over last 100 days
    
    Entry: CRSI < 10-15 = oversold (long), CRSI > 85-90 = overbought (short)
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_gain[max(0, i-streak_period+1):i+1])
        avg_loss = np.mean(streak_loss[max(0, i-streak_period+1):i+1])
        if avg_loss < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 3: Percentile Rank of returns
    returns = np.zeros(n)
    returns[0] = 0.0
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < returns[i]) / len(valid) * 100.0
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for weekly trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (use mean reversion)
        # CHOP < 38.2 = trending (use trend following)
        # 38.2 - 61.8 = neutral (use HTF bias only)
        
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === WEEKLY HTF BIAS ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_1d_fast[i]) and not np.isnan(hma_1d_fast[i-1]):
            if not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-1]):
                if hma_1d_fast[i-1] <= hma_1d[i-1] and hma_1d_fast[i] > hma_1d[i]:
                    hma_cross_long = True
                if hma_1d_fast[i-1] >= hma_1d[i-1] and hma_1d_fast[i] < hma_1d[i]:
                    hma_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (LOOSE: < 20 / > 80) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === ENTRY LOGIC (LOOSE - max 2-3 conditions for trade generation) ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY/RANGE (Connors RSI mean reversion)
        if is_choppy:
            # Long: CRSI < 20 + above SMA200 (just 2 conditions!)
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: CRSI > 80 + below SMA200 (just 2 conditions!)
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            
            # Extra: extreme CRSI alone for more trades
            elif crsi[i] < 10.0:
                desired_signal = SIZE_BASE
            elif crsi[i] > 90.0:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (breakout + HTF alignment)
        elif is_trending:
            # Long: Weekly bull + (Daily HMA bull OR Donchian breakout OR HMA cross)
            if htf_bull:
                if hma_bull or donchian_breakout_long or hma_cross_long:
                    desired_signal = SIZE_STRONG
            
            # Short: Weekly bear + (Daily HMA bear OR Donchian breakdown OR HMA cross)
            elif htf_bear:
                if hma_bear or donchian_breakdown_short or hma_cross_short:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 3: NEUTRAL (use HTF bias only, looser entries)
        else:
            # Long: Weekly bull + CRSI < 30 (mean reversion in neutral)
            if htf_bull and crsi[i] < 30.0:
                desired_signal = SIZE_BASE
            
            # Short: Weekly bear + CRSI > 70
            elif htf_bear and crsi[i] > 70.0:
                desired_signal = -SIZE_BASE
            
            # Or simple HMA cross with HTF agreement
            elif htf_bull and hma_cross_long:
                desired_signal = SIZE_BASE
            elif htf_bear and hma_cross_short:
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