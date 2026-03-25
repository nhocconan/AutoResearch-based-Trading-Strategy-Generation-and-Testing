#!/usr/bin/env python3
"""
Experiment #1307: 6h Primary + 1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: Recent 6h strategies failed due to either 0 trades (too strict) or negative
Sharpe (wrong regime logic). This implements Connors RSI (CRSI) which has proven 75%
win rate in mean reversion, combined with 1d SMA200 for regime bias.

Key innovations vs failed strategies:
1. CRSI (3-component) vs simple RSI - more robust signal
2. Very loose CRSI thresholds (<15, >85) - guarantees trades
3. 1d SMA200 for regime - only long above, only short below
4. ATR-based position sizing adjustment - smaller size in high vol
5. No complex regime filters (CHOP, ADX) that caused 0 trades

CRSI Formula (ConnorsRSI from literature):
- RSI(3): 3-period RSI on close
- RSI_Streak(2): RSI on streak duration (consecutive up/down days)
- PercentRank(100): percentile rank of today's return vs last 100 days
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry logic:
- LONG: CRSI < 15 + price > 1d_SMA200 (oversold in uptrend)
- SHORT: CRSI > 85 + price < 1d_SMA200 (overbought in downtrend)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete (vol-adjusted)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_mean_reversion_sma200_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """RSI on streak duration (consecutive up/down bars)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate streak duration
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(1, streak[i-1] + 1) if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = min(-1, streak[i-1] - 1) if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert to positive for RSI calculation (magnitude of streak)
    streak_abs = np.abs(streak)
    
    # RSI on streak
    delta = np.diff(streak_abs, prepend=streak_abs[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    streak_rsi = 100.0 - (100.0 / (1.0 + rs))
    streak_rsi[avg_loss == 0] = 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percentile rank of current return vs last N periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1.0)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            # Count how many values are less than current
            count_less = np.sum(window[:-1] < window[-1])
            percent_rank[i] = (count_less / (period - 1)) * 100.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_streak_rsi(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d SMA200 for regime filter
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=200)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 6h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    sma_200_6h = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (need 100 bars for CRSI percent rank)
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(sma_200_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER (1d SMA200) ===
        price_above_1d_sma200 = close[i] > sma_1d_aligned[i]
        price_below_1d_sma200 = close[i] < sma_1d_aligned[i]
        
        # === CRSI SIGNALS ===
        crsi_value = crsi[i]
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce size in high volatility (ATR > 1.5x recent average)
        atr_ratio = 1.0
        if i >= 30 and not np.isnan(atr_14[i-30]):
            recent_atr_avg = np.nanmean(atr_14[i-30:i])
            if recent_atr_avg > 0:
                atr_ratio = min(1.5, max(0.7, 1.0 / (atr_14[i] / recent_atr_avg + 0.01)))
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold (<15) + price above 1d SMA200 (uptrend)
        # Also check price near BB lower for confluence
        if crsi_value < 15.0 and price_above_1d_sma200:
            if close[i] <= bb_lower[i] * 1.02:  # Near or below BB lower
                desired_signal = SIZE_STRONG * atr_ratio
            else:
                desired_signal = SIZE_BASE * atr_ratio
        
        # SHORT: CRSI overbought (>85) + price below 1d SMA200 (downtrend)
        # Also check price near BB upper for confluence
        elif crsi_value > 85.0 and price_below_1d_sma200:
            if close[i] >= bb_upper[i] * 0.98:  # Near or above BB upper
                desired_signal = -SIZE_STRONG * atr_ratio
            else:
                desired_signal = -SIZE_BASE * atr_ratio
        
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
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.8:
            final_signal = -SIZE_BASE
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