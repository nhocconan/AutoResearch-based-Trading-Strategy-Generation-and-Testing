#!/usr/bin/env python3
"""
Experiment #1387: 6h Primary + 1d/1w HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: 6h timeframe is underexplored. This strategy uses Connors RSI (CRSI) for 
mean-reversion entries filtered by 1d HMA trend direction. CRSI has proven 75%+ win 
rate in academic studies and works well in bear/range markets (2022-2025).

Why this should work where others failed:
- CRSI catches oversold/overbought extremes better than standard RSI
- 1d HMA filter prevents counter-trend trades that destroyed Sharpe in 2022
- 6h TF = natural 30-50 trades/year (fee-friendly)
- Mean-reversion works in both bull and bear markets (unlike pure trend-following)
- Loose entry thresholds guarantee trades (CRSI<20 or >80, not <10 or >90)

Entry logic:
- LONG: CRSI < 20 + price > 1d_HMA(21) + 1w_HMA confirming
- SHORT: CRSI > 80 + price < 1d_HMA(21) + 1w_HMA confirming

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_meanreversion_hma_trend_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak Component for Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate streak values
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak
    if n >= period + 1:
        delta = np.diff(streak)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain = np.insert(gain, 0, 0)
        loss = np.insert(loss, 0, 0)
        
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        mask = avg_loss != 0
        rs = np.zeros(n)
        rs[mask] = avg_gain[mask] / avg_loss[mask]
        streak_rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank for Connors RSI
    Measures where current return ranks vs recent history
    """
    n = len(close)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate returns
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    # Calculate percent rank
    for i in range(period, n):
        window = returns[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current = returns[i]
            rank = np.sum(valid <= current)
            percent_rank[i] = rank / len(valid) * 100
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major regime confirmation
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_value = crsi[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + 1d bullish trend
        # Use CRSI < 25 (not <10) to ensure we get trades
        if crsi_value < 25 and price_above_1d:
            if price_above_1w:
                # Strong trend alignment (1d + 1w both bullish)
                desired_signal = SIZE_LONG
            else:
                # Basic long (only 1d bullish)
                desired_signal = SIZE_LONG * 0.8
        
        # SHORT: CRSI overbought + 1d bearish trend
        # Use CRSI > 75 (not >90) to ensure we get trades
        elif crsi_value > 75 and price_below_1d:
            if price_below_1w:
                # Strong trend alignment (1d + 1w both bearish)
                desired_signal = -SIZE_SHORT
            else:
                # Basic short (only 1d bearish)
                desired_signal = -SIZE_SHORT * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= -SIZE_SHORT * 0.9:
            final_signal = -SIZE_SHORT
        elif desired_signal >= SIZE_LONG * 0.5:
            final_signal = SIZE_LONG * 0.8
        elif desired_signal <= -SIZE_SHORT * 0.5:
            final_signal = -SIZE_SHORT * 0.8
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