#!/usr/bin/env python3
"""
Experiment #344: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime v1

Hypothesis: Previous strategies failed due to TOO MANY conflicting filters causing 0 trades.
This strategy SIMPLIFIES entry logic using proven Connors RSI (75% win rate in literature)
with Choppiness Index regime detection. Key changes from #324:

1. CONNORS RSI instead of standard RSI - better mean reversion signal
   CRSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
2. SIMPLER regime: CHOP > 58 = choppy (CRSI mean revert), CHOP < 45 = trending (HMA breakout)
3. LOOSENED entries: CRSI < 25 or > 75 (not 10/90) to ensure 30+ trades/year
4. REDUCED filters: Only require 2-3 confirmations (not 5-6) to avoid 0 trades
5. Single HTF filter (1d HMA) instead of multiple (1d + 1w) to reduce conflicts

Entry Logic:
- Choppy regime: CRSI < 25 + price > SMA100 → long; CRSI > 75 + price < SMA100 → short
- Trending regime: HMA(21) crossover + 1d HMA alignment → enter with trend
- Position size: 0.25 base, 0.30 when 1d HTF confirms direction

Target: Sharpe > 0.45, trades >= 30 train, trades >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_simplified_v1"
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

def calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion
    CRSI = (RSI(close, 2) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Literature shows 75% win rate for CRSI < 10 (long) and CRSI > 90 (short)
    We use < 25 and > 75 for more trades
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # Component 1: RSI(2) of price
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    # Create "gains" when streak is positive (up streak), "losses" when negative
    streak_gain = np.where(streak > 0, streak_abs, 0.0)
    streak_loss = np.where(streak < 0, streak_abs, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
    
    # Component 3: Percentile Rank of close over lookback
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i]) / rank_period * 100.0
        
        # Combine all 3 components
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + rank) / 3.0
    
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
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100)
    sma_100 = calculate_sma(close, 100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 58.0
        trending_threshold = 45.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d only - simplified) ===
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
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === CRSI EXTREMES (LOOSENED for more trades) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (CRSI mean reversion)
        if current_regime == 2:
            # Long: CRSI oversold + above SMA100 (trend filter)
            if crsi_oversold and above_sma100:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: CRSI overbought + below SMA100
            elif crsi_overbought and below_sma100:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (HMA breakout with HTF confirmation)
        elif current_regime == 1:
            # Long: HMA bull + HMA cross + 1d bull
            if hma_bull and hma_cross_long and htf_1d_bull:
                desired_signal = SIZE_STRONG
            
            # Short: HMA bear + HMA cross + 1d bear
            elif hma_bear and hma_cross_short and htf_1d_bear:
                desired_signal = -SIZE_STRONG
            
            # Simpler: just HMA direction + HTF alignment
            elif hma_bull and htf_1d_bull:
                desired_signal = SIZE_BASE
            
            elif hma_bear and htf_1d_bear:
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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