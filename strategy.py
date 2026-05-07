#!/usr/bin/env python3
# 1D_KAMA_Trend_RSI_ChopFilter_v1
# Hypothesis: Daily strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
# combined with RSI for momentum confirmation and Choppiness Index for regime filtering.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# RSI filters for overbought/oversold conditions within the trend.
# Choppiness Index (CHOP > 61.8) identifies ranging markets where we avoid trend trades.
# Designed to work in both bull and bear markets by only taking trend-aligned trades.
# Targets 15-25 trades/year to minimize fee drag on daily timeframe.

name = "1D_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER length = 10, Fast SC = 2/(2+1), Slow SC = 2/(30+1)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will be corrected below
    
    # Correct efficiency ratio calculation
    er = np.zeros_like(close)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start KAMA at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate True Range for Choppiness
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) for Choppiness denominator
    atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_raw = np.full_like(df_1w['high'], np.nan)
    for i in range(13, len(df_1w)):
        atr_sum = np.sum(atr14[i-13:i+1])
        max_high = np.max(df_1w['high'].values[i-13:i+1])
        min_low = np.min(df_1w['low'].values[i-13:i+1])
        if max_high > min_low and atr_sum > 0:
            chop_raw[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_raw[i] = 50  # Neutral value
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama, additional_delay_bars=0)  # KAMA uses same TF
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi, additional_delay_bars=0)   # RSI uses same TF
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw, additional_delay_bars=0)  # CHOP from weekly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 10)  # Ensure we have RSI and KAMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI > 50 (bullish momentum) + Chop < 61.8 (trending market)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI < 50 (bearish momentum) + Chop < 61.8 (trending market)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price crosses KAMA (trend change)
            # 2. RSI reaches extreme levels (overbought/oversold)
            kama_cross = (position == 1 and close[i] < kama_aligned[i]) or \
                         (position == -1 and close[i] > kama_aligned[i])
            rsi_extreme = (position == 1 and rsi_aligned[i] >= 70) or \
                          (position == -1 and rsi_aligned[i] <= 30)
            
            if kama_cross or rsi_extreme:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals