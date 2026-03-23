#!/usr/bin/env python3
"""
Experiment #1411: 4h Primary + 1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Connors RSI (CRSI) has documented 75% win rate for mean reversion entries.
Combined with 1d HMA trend filter, this should:
1. Enter long on pullbacks in uptrend (CRSI<20 + price>1d_HMA)
2. Enter short on rallies in downtrend (CRSI>80 + price<1d_HMA)
3. Generate 30-50 trades/year on 4h timeframe
4. Work across BTC/ETH/SOL (mean reversion is universal)

Why this should beat #1404 (Sharpe=0.093):
- #1404 had TOO MANY confluence filters (Donchian + HMA + KAMA + RSI + ATR) = rare entries
- CRSI is a SINGLE composite indicator with proven edge
- Simpler entry = more trades = better statistics
- 1d HMA filter prevents counter-trend mean reversion (the killer in 2022)

Design:
1. 1d HMA(21) = macro trend filter (long only above, short only below)
2. 4h Connors RSI(3,2,100) = entry trigger (extreme values = mean reversion)
3. 4h ATR(14) = trailing stop 2.5x (mandatory risk management)
4. Position size 0.28 = conservative for 4h volatility
5. Exit when CRSI crosses back through 50 (mean reached)

Target: Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 5 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Streak RSI component of Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    for i in range(period, n):
        window = streak[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            gains = np.where(window > 0, window, 0)
            losses = np.where(window < 0, -window, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss > 1e-10:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            else:
                streak_rsi[i] = 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI.
    Measures where current price is relative to last N periods.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            current = close[i]
            count_below = np.sum(window[:-1] < current)
            pr[i] = (count_below / (period - 1)) * 100.0
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    rsi_short = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_streak_rsi(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    crsi = np.full(len(close), np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
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
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - strongest filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI EXTREMES (mean reversion signals) ===
        # CRSI < 20 = oversold (long opportunity in uptrend)
        # CRSI > 80 = overbought (short opportunity in downtrend)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # CRSI crossing back through 50 = exit signal (mean reached)
        crsi_neutral = 40.0 < crsi[i] < 60.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Uptrend + CRSI oversold
        if macro_bull and crsi_oversold:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Downtrend + CRSI overbought
        elif macro_bear and crsi_overbought:
            desired_signal = -BASE_SIZE
        
        # EXIT: CRSI returned to neutral (mean reversion complete)
        if in_position and crsi_neutral:
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
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE
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