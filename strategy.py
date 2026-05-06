#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily KAMA trend with RSI momentum and chop filter
# KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets
# RSI(14) filters for momentum strength (>50 for long, <50 for short) to avoid choppy entries
# Choppiness Index (CHOP) > 61.8 avoids trend-following in ranging markets, reducing whipsaw
# Works in bull markets via trend continuation, in bear markets via counter-trend bounces at extremes
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_KAMA_RSI_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily KAMA trend ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Kaufman's Adaptive Moving Average (KAMA)
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(df_1d['close'].diff(10))
    volatility = df_1d['close'].diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = df_1d['close'].copy()
    for i in range(1, len(kama)):
        if not np.isnan(sc.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    kama_values = kama.values
    
    # Align daily KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # RSI(14) on daily for momentum filter
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Choppiness Index (CHOP) on 12h for regime filter
    # Requires high, low, close
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean().values
    
    # Calculate highest high and lowest low over atr_period
    highest_high = pd.Series(high).rolling(atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_sum = highest_high - lowest_low
    chop = np.where(
        (range_sum > 0) & (atr > 0),
        100 * np.log10(atr * atr_period / range_sum) / np.log10(atr_period),
        50  # neutral when range is zero
    )
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, and not in strong chop (trending market)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and chop[i] <= 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, and not in strong chop
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and chop[i] <= 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or RSI < 40 (loss of momentum)
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or RSI > 60 (loss of momentum)
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals