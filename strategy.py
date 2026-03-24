#!/usr/bin/env python3
"""
Experiment #498: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: 4h timeframe with 1d HTF provides optimal balance between signal frequency
and noise reduction. Connors RSI (CRSI) has proven 75% win rate for mean reversion.
Choppiness Index filters between trending (follow trend) and ranging (mean revert) regimes.
This dual-regime approach should work in both bull and bear markets.

Strategy logic:
1. 1d HMA(21) = daily trend bias (HTF filter)
2. 4h Choppiness(14) = regime detection (>61.8 = range, <38.2 = trend)
3. 4h Connors RSI = entry timing (CRSI<10 long, CRSI>90 short)
4. 4h HMA(21) = short-term trend confirmation
5. ATR(14)*2.5 stoploss on all positions
6. Regime-adaptive: mean revert in chop, trend follow otherwise

Key innovations:
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Choppiness Index = 100 * (ATR(14) / (Highest High - Lowest Low)) * 100
- Dual regime logic adapts to market conditions
- Loose CRSI thresholds (15/85 not 10/90) to ensure trade generation

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=12 test
Timeframe: 4h (proven higher TF works best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_1d_v1"
timeframe = "4h"
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
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * (ATR(14) / (Highest High - Lowest Low over 14)) * 100
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if price_range[i] > 1e-10 and atr[i] > 1e-10:
            chop[i] = 100.0 * (atr[i] / price_range[i]) * np.log10(period) / np.log10(2)
            chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_abs = np.abs(streak[max(0, i-streak_period):i+1])
        if len(streak_abs) > 0:
            avg_streak = np.mean(streak_abs)
            streak_rsi[i] = 100.0 * avg_streak / (avg_streak + 1.0)
            if streak[i] < 0:
                streak_rsi[i] = 100.0 - streak_rsi[i]
    
    # Percent Rank of returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window_returns = returns[max(0, i-rank_period):i]
        if len(window_returns) > 0:
            count_below = np.sum(window_returns < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window_returns)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (mean revert)
        is_trending = chop[i] < 45.0  # Trend market (trend follow)
        
        # === CONNORS RSI EXTREMES (LOOSE: 15/85 for trade generation) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 3.0  # Avoid extreme vol spikes
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow HTF trend with CRSI pullback entries
        if is_trending and vol_normal:
            # Long: 1d bull + 4h HMA bull + CRSI pullback
            if htf_bull and hma_bull and above_sma50:
                if crsi_oversold and rsi_oversold:
                    desired_signal = SIZE_STRONG
                elif crsi[i] < 30.0 and rsi_14[i] < 45.0:
                    desired_signal = SIZE_BASE
                elif crsi[i] < 40.0 and hma_bull:
                    desired_signal = SIZE_BASE * 0.8
            
            # Short: 1d bear + 4h HMA bear + CRSI rally
            elif htf_bear and hma_bear and below_sma50:
                if crsi_overbought and rsi_overbought:
                    desired_signal = -SIZE_STRONG
                elif crsi[i] > 70.0 and rsi_14[i] > 55.0:
                    desired_signal = -SIZE_BASE
                elif crsi[i] > 60.0 and hma_bear:
                    desired_signal = -SIZE_BASE * 0.8
        
        # CHOPPY REGIME: Mean reversion at extremes
        elif is_choppy and vol_normal:
            # Long: CRSI extreme oversold + above SMA200
            if crsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif crsi_oversold and above_sma50 and rsi_oversold:
                desired_signal = SIZE_BASE * 0.8
            
            # Short: CRSI extreme overbought + below SMA200
            elif crsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            elif crsi_overbought and below_sma50 and rsi_overbought:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME: Pure CRSI mean reversion
        if desired_signal == 0.0 and vol_normal:
            if crsi_extreme_oversold and above_sma50:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_extreme_overbought and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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