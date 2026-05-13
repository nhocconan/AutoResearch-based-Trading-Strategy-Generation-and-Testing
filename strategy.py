#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend + RSI mean reversion + volume spike + chop regime filter
# Long when KAMA rising (bullish trend), RSI < 30 (oversold), volume > 2.0x 20-day average, and choppy market (CHOP > 61.8)
# Short when KAMA falling (bearish trend), RSI > 70 (overbought), volume > 2.0x 20-day average, and choppy market (CHOP > 61.8)
# Uses 1d primary timeframe and 1w HTF for regime alignment (choppiness index on 1w)
# Designed for BTC/ETH to work in both bull and bear markets by combining trend-following entry with mean reversion in choppy conditions
# Volume spike confirms authenticity of reversion signal
# Discrete position sizing (0.25) to minimize fee churn
# ATR-based trailing stop (2.5x) for risk management

name = "1d_KAMA_RSI_Volume_Chop_Regime_v1"
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
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # Sum of |close[t] - close[t-1]| over 10 periods
    # Pad arrays to match length
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first ER calculation
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # Get 1w data for Choppiness Index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index(14) on 1w data
    # True Range
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]  # First bar
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).sum().values  # Sum of TR over 14 periods
    
    # Highest high and lowest low over 14 periods
    hh_w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_w / (hh_w - ll_w)) / log10(14)
    # Avoid division by zero
    range_w = hh_w - ll_w
    chop_w = np.where(range_w > 0, 100 * np.log10(atr_w / range_w) / np.log10(14), 50)
    chop_w = np.where(np.isnan(chop_w), 50, chop_w)  # Default to 50 if undefined
    
    # Align HTF arrays to 1d timeframe (wait for completed 1w bar)
    chop_w_aligned = align_htf_to_ltf(prices, df_1w, chop_w)
    chop_filter = chop_w_aligned > 61.8  # Choppy market regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (bullish trend), RSI < 30 (oversold), volume spike, choppy market
            if kama[i] > kama[i-1] and rsi[i] < 30 and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: KAMA falling (bearish trend), RSI > 70 (overbought), volume spike, choppy market
            elif kama[i] < kama[i-1] and rsi[i] > 70 and volume_filter[i] and chop_filter[i]:
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