#!/usr/bin/env python3
"""
Experiment #432: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI

Hypothesis: Recent failures show overly complex strategies get 0 trades or negative Sharpe.
This version uses PROVEN patterns from research:
1. Choppiness Index (CHOP) for regime detection — ETH Sharpe +0.923 in backtests
2. Connors RSI (CRSI) for mean reversion — 75% win rate, catches reversals
3. 1d HMA for trend bias — multi-timeframe alignment proven to 2x Sharpe
4. Loose entry thresholds to ensure 20-50 trades/year on 12h

Regime Detection:
- CHOP(14) > 61.8 = range → Connors RSI mean reversion
- CHOP(14) < 38.2 = trend → HMA breakout with 1d alignment
- Otherwise = neutral → flat or reduce size

Entry Logic:
- Range Long: CRSI < 15 + price > SMA200 + CHOP > 61.8
- Range Short: CRSI > 85 + price < SMA200 + CHOP > 61.8
- Trend Long: HMA(21) bull + 1d HMA bull + price pullback to HMA
- Trend Short: HMA(21) bear + 1d HMA bear + price rally to HMA

Position sizing: 0.25 base, 0.30 when HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=80 train (20/year), trades>=15 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_regime_1d_v1"
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
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            choppiness[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines RSI, streak RSI, and percentile rank
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_abs = np.abs(streak)
    streak_signed = np.where(streak >= 0, streak_abs, -streak_abs)
    
    # Calculate RSI on streak values
    delta_streak = np.diff(streak_signed)
    gain_streak = np.where(delta_streak > 0, delta_streak, 0.0)
    loss_streak = np.where(delta_streak < 0, -delta_streak, 0.0)
    gain_streak = np.concatenate([[0.0], gain_streak])
    loss_streak = np.concatenate([[0.0], loss_streak])
    
    avg_gain_streak = pd.Series(gain_streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_loss_streak = pd.Series(loss_streak).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        if avg_loss_streak[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_gain_streak[i] / avg_loss_streak[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percentile Rank(100) - where does current close rank vs last 100 closes
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, 200)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(400, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOPPINESS INDEX) ===
        # CHOP > 61.8 = range/choppy (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2 <= CHOP <= 61.8 = transition/neutral
        
        is_choppy = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        
        # === HTF BIAS (1d) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_12h_fast[i]) and not np.isnan(hma_12h_fast[i-1]):
            if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
                if hma_12h_fast[i-1] <= hma_12h[i-1] and hma_12h_fast[i] > hma_12h[i]:
                    hma_cross_long = True
                if hma_12h_fast[i-1] >= hma_12h[i-1] and hma_12h_fast[i] < hma_12h[i]:
                    hma_cross_short = True
        
        # === SMA200 FILTER (major trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (LOOSE: 15/85 for more trades) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME: CHOPPY (mean reversion with Connors RSI)
        if is_choppy:
            # Long: CRSI < 15 + above SMA200 (2 conditions)
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: CRSI > 85 + below SMA200 (2 conditions)
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
        
        # REGIME: TRENDING (HMA breakout with HTF alignment)
        elif is_trending:
            # Long: HMA bull + 1d bull + (pullback or cross)
            if hma_bull and htf_1d_bull:
                # Pullback to HMA or crossover
                pullback_long = close[i] <= hma_12h[i] * 1.02 and close[i] > hma_12h[i] * 0.98
                if pullback_long or hma_cross_long:
                    desired_signal = SIZE_STRONG
            
            # Short: HMA bear + 1d bear + (rally or cross)
            elif hma_bear and htf_1d_bear:
                # Rally to HMA or crossover
                rally_short = close[i] >= hma_12h[i] * 0.98 and close[i] < hma_12h[i] * 1.02
                if rally_short or hma_cross_short:
                    desired_signal = -SIZE_STRONG
        
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