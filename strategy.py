#!/usr/bin/env python3
"""
Experiment #040: 6h Primary + 1d/1w HTF — Connors RSI Mean Reversion + HMA Trend

Hypothesis: 6h timeframe is underexplored (ZERO successful experiments). After 39 failures,
the pattern shows complex regime filters (Choppiness, Fisher) fail on 6h. SOLUTION:
- Connors RSI (CRSI) has proven 75% win rate for mean reversion in literature
- 6h candles are long enough that extreme CRSI readings are meaningful (not noise)
- 1d HMA(50) provides trend bias without being too restrictive
- 1w HMA(21) filters only extreme counter-trend cases
- LOOSE CRSI thresholds (25/75 not 15/85) to ensure >=30 trades on train
- This is DIFFERENT from all 39 failed experiments (no Choppiness, no Fisher, no Donchian)

Key design choices:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1d HMA(50) + 1w HMA(21) for trend bias
- Entry: CRSI < 25 for long (with 1d bull), CRSI > 75 for short (with 1d bear)
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Position size: 0.30 (30% of capital)
- Stoploss: 2.5x ATR trailing
- LOOSE filters to ensure trades generate on ALL symbols

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_hma_dual_htf_v1"
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
    Connors RSI Streak Component
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(5, n):
        streak = 0
        if i >= 1:
            if close[i] > close[i-1]:
                streak = 1
                for j in range(i-1, 0, -1):
                    if close[j] > close[j-1]:
                        streak += 1
                    else:
                        break
            elif close[i] < close[i-1]:
                streak = -1
                for j in range(i-1, 0, -1):
                    if close[j] < close[j-1]:
                        streak -= 1
                    else:
                        break
        
        # Calculate RSI of streak values over lookback
        streak_values = []
        for k in range(i-period, i+1):
            if k < 1:
                continue
            s = 0
            if close[k] > close[k-1]:
                s = 1
                for m in range(k-1, max(0, k-period-1), -1):
                    if close[m] > close[m-1]:
                        s += 1
                    else:
                        break
            elif close[k] < close[k-1]:
                s = -1
                for m in range(k-1, max(0, k-period-1), -1):
                    if close[m] < close[m-1]:
                        s -= 1
                    else:
                        break
            streak_values.append(s)
        
        if len(streak_values) >= period:
            streak_arr = np.array(streak_values[-period:])
            gain_streak = np.where(streak_arr > 0, streak_arr, 0.0)
            loss_streak = np.where(streak_arr < 0, -streak_arr, 0.0)
            avg_gain = np.mean(gain_streak)
            avg_loss = np.mean(loss_streak)
            if avg_loss < 1e-10:
                streak_rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank - where current close ranks vs last period closes
    0 = lowest, 100 = highest
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
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
        if np.isnan(crsi[i]):
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
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        # LOOSE thresholds to ensure trades generate (25/75 not 15/85)
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + 1d HMA bull (trend-aligned mean reversion)
        # Also allow if 1w strongly bull (overrules 1d)
        if crsi_oversold:
            if htf_1d_bull:
                desired_signal = SIZE
            elif htf_1w_bull and crsi[i] < 20.0:
                # Very oversold + weekly bull = strong long
                desired_signal = SIZE
        
        # SHORT: CRSI overbought + 1d HMA bear (trend-aligned mean reversion)
        # Also allow if 1w strongly bear (overrules 1d)
        if crsi_overbought:
            if htf_1d_bear:
                desired_signal = -SIZE
            elif htf_1w_bear and crsi[i] > 80.0:
                # Very overbought + weekly bear = strong short
                desired_signal = -SIZE
        
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