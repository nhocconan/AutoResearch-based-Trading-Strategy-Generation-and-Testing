#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction + RSI(14) mean reversion + choppiness regime filter.
# Long when KAMA is rising (bullish regime), RSI < 30 (oversold), and choppy market (CHOP > 61.8).
# Short when KAMA is falling (bearish regime), RSI > 70 (overbought), and choppy market (CHOP > 61.8).
# Uses ATR(14) trailing stop (2.5x) for risk control.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.

name = "1d_KAMA_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, n=1))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(change) > 0 else np.array([])
    # For ER calculation, we need to compute rolling volatility
    change_series = pd.Series(np.diff(close, n=1))
    volatility_sum = change_series.rolling(window=10, min_periods=10).sum().values
    change_magnitude = np.abs(np.diff(close, n=1))
    er = np.zeros_like(close)
    er[10:] = change_magnitude[9:] / np.where(volatility_sum[9:] == 0, 1, volatility_sum[9:])
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50  # Neutral start
    
    # Calculate Choppiness Index (CHOP)
    # True Range
    tr_chop = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_chop[0] = high[0] - low[0]
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr_chop).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.zeros_like(close)
    chop[13:] = 100 * np.log10(tr_sum[13:] / (hh[13:] - ll[13:])) / np.log10(14)
    # Handle edge cases where hh == ll
    chop = np.where((hh - ll) == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(14, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (bullish), RSI < 30 (oversold), choppy market (CHOP > 61.8)
            if kama[i] > kama[i-1] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: KAMA falling (bearish), RSI > 70 (overbought), choppy market (CHOP > 61.8)
            elif kama[i] < kama[i-1] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals