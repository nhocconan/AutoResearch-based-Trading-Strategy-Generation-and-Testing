#!/usr/bin/env python3
"""
Experiment #102: 12h Primary + 1d HTF — Connors RSI Mean Reversion + HMA Trend

Hypothesis: After 100+ failed experiments, clear patterns emerge for 12h timeframe:
- Pure trend following fails on BTC/ETH (2022 crash whipsaw)
- Pure mean reversion fails without trend filter (catches falling knives)
- Connors RSI has proven 75% win rate in research literature
- 12h timeframe should generate 20-50 trades/year (lower than 4h's 50-100)

This strategy combines:
1. 1d HMA = major trend bias (price above/below for long/short bias)
2. Connors RSI (3-period RSI + 2-period streak RSI + 100-bar percentile rank)
3. 12h HMA(16) crossover for entry timing precision
4. ATR(14) trailing stop at 2.5x for risk management
5. Mean reversion exit when CRSI crosses 50 (capture quick reversals)

Key design choices:
- Timeframe: 12h (instruction requirement, proven higher TF works)
- HTF: 1d for trend bias (responsive enough, not too noisy like 1w)
- Connors RSI thresholds: 25/75 (looser than classic 10/90 to ensure trades)
- Position size: 0.28 (28% of capital, conservative for 12h volatility)
- Stoploss: 2.5x ATR trailing (tighter than 3x for better risk/reward)
- Discrete signals: 0.0, ±0.28 only (minimize fee churn)

Target: Sharpe>0.351 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_meanrev_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=16):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(prices, span):
        result = np.full(len(prices), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(prices)):
            window = prices[i - span + 1:i + 1]
            result[i] = np.sum(weights * window) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + rsi_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3) component
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2) component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / (avg_streak_loss[i] + 1e-10)
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank(100) component
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period)
        crsi[i] = (rsi_short[i] + rsi_streak[i] + rank * 100.0) / 3.0
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 12h)
    
    # Position tracking for stoploss and exit logic
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h MOMENTUM (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        # LOOSE thresholds to ensure trades generate on all symbols
        crsi_oversold = crsi[i] < 35.0  # Was 25, loosened for more trades
        crsi_overbought = crsi[i] > 65.0  # Was 75, loosened for more trades
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 12h HMA cross bull + CRSI oversold
        # SHORT: 1d bear + 12h HMA cross bear + CRSI overbought
        desired_signal = 0.0
        
        if htf_bull and hma_cross_bull and crsi_oversold:
            desired_signal = SIZE
        elif htf_bear and hma_cross_bear and crsi_overbought:
            desired_signal = -SIZE
        
        # === MEAN REVERSION EXIT (CRSI crosses 50) ===
        # Close longs when CRSI rises above 50, close shorts when CRSI falls below 50
        if in_position and position_side > 0 and crsi[i] > 50.0:
            desired_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 50.0:
            desired_signal = 0.0
        
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