#!/usr/bin/env python3
"""
Experiment #1336: 12h Primary + 1d HTF — Connors RSI Mean Reversion + HMA Trend Filter

Hypothesis: Connors RSI (CRSI) is a proven mean-reversion indicator with 75% win rate
in academic literature. Combined with 1d HMA for trend bias and ATR stoploss, this
should work in both bull and bear markets. 12h timeframe targets 20-50 trades/year
to minimize fee drag while maintaining sufficient trade count.

Key components:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. 1d HMA(21) for macro trend bias (long only above, short only below)
3. Entry: CRSI < 15 for long, CRSI > 85 for short (wider than standard for more trades)
4. ATR(14) trailing stop at 2.5x for risk management
5. Position size: 0.28 (discrete, conservative)

Target: 25-50 trades/year on 12h, Sharpe > 0.612, trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi_fast(close, period=3):
    """RSI with fast period for Connors RSI"""
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

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate streak: +1 for up, -1 for down, 0 for flat
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.full(n, np.nan)
    for i in range(period, n):
        abs_streak = abs(streak[i])
        if streak[i] > 0:
            streak_rsi[i] = 50.0 + (abs_streak / period) * 50.0
        elif streak[i] < 0:
            streak_rsi[i] = 50.0 - (abs_streak / period) * 50.0
        else:
            streak_rsi[i] = 50.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Smooth with EMA
    streak_rsi = pd.Series(streak_rsi).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures current price change vs past period changes
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate daily returns
    returns = np.diff(close, prepend=close[0]) / (close[0] + 1e-10)
    returns[0] = 0.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            current = returns[i]
            count_below = np.sum(window[:-1] < current)
            percent_rank[i] = (count_below / (period - 1)) * 100.0
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with 75% win rate
    """
    rsi_fast = calculate_rsi_fast(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(len(close), np.nan)
    for i in range(pr_period, len(close)):
        if not np.isnan(rsi_fast[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # 12h HMA for local trend
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS ===
        # Long: CRSI extremely oversold (< 15) + trend confirmation
        long_signal = False
        if macro_bull and local_bull and above_sma200:
            # Strong trend: enter on moderate oversold
            if crsi[i] < 25.0:
                long_signal = True
        elif macro_bull or local_bull:
            # Weaker trend: require extreme oversold
            if crsi[i] < 15.0:
                long_signal = True
        
        # Short: CRSI extremely overbought (> 85) + trend confirmation
        short_signal = False
        if macro_bear and local_bear and below_sma200:
            # Strong trend: enter on moderate overbought
            if crsi[i] > 75.0:
                short_signal = True
        elif macro_bear or local_bear:
            # Weaker trend: require extreme overbought
            if crsi[i] > 85.0:
                short_signal = True
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if long_signal:
            desired_signal = BASE_SIZE
        elif short_signal:
            desired_signal = -BASE_SIZE
        
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
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI becomes overbought, exit short when oversold
        if in_position and desired_signal == 0.0:
            if position_side > 0 and crsi[i] > 70.0:
                desired_signal = 0.0  # Take profit on long
            elif position_side < 0 and crsi[i] < 30.0:
                desired_signal = 0.0  # Take profit on short
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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