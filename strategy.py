#!/usr/bin/env python3
"""
Experiment #618: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Connors RSI

Hypothesis: Complex regime detection (ADX+Choppiness+KAMA) in #522 is too restrictive,
causing 0 trades. Simplify to proven patterns: 1d HMA for macro bias, 4h HMA for trend,
RSI pullback for entries. Add Connors RSI for mean reversion when 1d trend is neutral.

Key changes from failed #522:
1. Removed KAMA - use simpler HMA(9/18) crossover (proven to work)
2. Removed ADX/Choppiness regime - too restrictive, causes 0 trades
3. Relaxed RSI thresholds: <45/>55 instead of <35/>65 (more trade opportunities)
4. Added Connors RSI for additional mean reversion signals
5. Simpler logic: 1d trend direction + 4h pullback entry
6. Ensure minimum trade frequency: 30-60 trades/year target

Strategy logic:
1. 1d HMA(21) = macro trend bias (long only if price>1d_HMA, short if price<1d_HMA)
2. 4h HMA(9) vs HMA(18) = trend confirmation (fast>slow = bull)
3. RSI(14) pullback: enter long when RSI<45 in uptrend, short when RSI>55 in downtrend
4. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Entry when CRSI<15 (oversold) or CRSI>85 (overbought)
5. ATR(14)*2.5 stoploss on all positions
6. Discrete sizing: 0.0, ±0.25, ±0.30

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_crsi_simplified_1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_changes = np.zeros(streak_period)
        for j in range(streak_period):
            if i - j > 0:
                streak_changes[j] = 1 if streak[i-j] > 0 else (0 if streak[i-j] == 0 else -1)
        
        gains = np.sum(np.where(streak_changes > 0, streak_changes, 0))
        losses = np.abs(np.sum(np.where(streak_changes < 0, streak_changes, 0)))
        
        if losses < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = gains / losses
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            rank = np.sum(returns[:-1] < returns[-1])
            percent_rank[i] = 100.0 * rank / len(returns[:-1])
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_9 = calculate_hma(close, period=9)
    hma_18 = calculate_hma(close, period=18)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_9[i]) or np.isnan(hma_18[i]) or np.isnan(rsi_14[i]):
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
        
        # === 1D MACRO TREND BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bull = hma_9[i] > hma_18[i]
        hma_bear = hma_9[i] < hma_18[i]
        
        # HMA slope confirmation
        hma_slope_bull = hma_9[i] > hma_9[i-5] if i >= 5 and not np.isnan(hma_9[i-5]) else False
        hma_slope_bear = hma_9[i] < hma_9[i-5] if i >= 5 and not np.isnan(hma_9[i-5]) else False
        
        # === RSI PULLBACK ===
        rsi_pullback_long = rsi_14[i] < 45.0
        rsi_pullback_short = rsi_14[i] > 55.0
        rsi_extreme_long = rsi_14[i] < 35.0
        rsi_extreme_short = rsi_14[i] > 65.0
        
        # === CONNORS RSI ===
        crsi_oversold = not np.isnan(crsi[i]) and crsi[i] < 20.0
        crsi_overbought = not np.isnan(crsi[i]) and crsi[i] > 80.0
        crsi_extreme_oversold = not np.isnan(crsi[i]) and crsi[i] < 15.0
        crsi_extreme_overbought = not np.isnan(crsi[i]) and crsi[i] > 85.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES (1d bull + 4h pullback)
        if htf_bull:
            # Standard pullback entry
            if hma_bull and rsi_pullback_long:
                desired_signal = SIZE_BASE
            # Strong entry with CRSI confirmation
            if hma_bull and hma_slope_bull and crsi_extreme_oversold:
                desired_signal = SIZE_STRONG
            # RSI extreme in uptrend
            if hma_bull and rsi_extreme_long:
                desired_signal = SIZE_STRONG
            # CRSI oversold alone (mean reversion)
            if crsi_extreme_oversold and rsi_14[i] < 40.0:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES (1d bear + 4h pullback)
        elif htf_bear:
            # Standard pullback entry
            if hma_bear and rsi_pullback_short:
                desired_signal = -SIZE_BASE
            # Strong entry with CRSI confirmation
            if hma_bear and hma_slope_bear and crsi_extreme_overbought:
                desired_signal = -SIZE_STRONG
            # RSI extreme in downtrend
            if hma_bear and rsi_extreme_short:
                desired_signal = -SIZE_STRONG
            # CRSI overbought alone (mean reversion)
            if crsi_extreme_overbought and rsi_14[i] > 60.0:
                desired_signal = -SIZE_BASE
        
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
                entry_atr = atr_14[i]
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