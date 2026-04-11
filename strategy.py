#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: Daily KAMA trend with RSI momentum and Choppiness index regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: In both bull and bear markets, price tends to trend when Choppiness Index is low (<38.2).
# KAMA adapts to trend strength and noise, providing reliable trend signals.
# RSI filters for momentum exhaustion to avoid chasing extremes.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (<25/year) to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly close for trend filter
    close_1w = df_1w['close'].values
    
    # KAMA parameters
    fast_sc = 0.666
    slow_sc = 0.0645
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 9 values
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with first close
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    # Wilder smoothing
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr_1 = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_1[0] = high[0] - low[0]  # First value
    atr_sum = np.nancumsum(atr_1)  # Cumulative sum for rolling sum
    atr_sum_14 = np.where(np.arange(len(atr_sum)) >= 13, 
                          atr_sum - np.roll(atr_sum, 14), 
                          np.nan)
    atr_sum_14[:13] = np.nan
    # True range for first 13 values
    for i in range(1, 14):
        if i < len(atr_sum_14):
            atr_sum_14[i] = np.sum(atr_1[:i+1])
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    highest_high_14 = np.where(np.arange(len(highest_high)) >= 13, 
                               highest_high - np.roll(highest_high, 14), 
                               highest_high)
    lowest_low_14 = np.where(np.arange(len(lowest_low)) >= 13, 
                             lowest_low - np.roll(lowest_low, 14), 
                             lowest_low)
    for i in range(13):
        highest_high_14[i] = np.max(high[:i+1])
        lowest_low_14[i] = np.min(low[:i+1])
    chop = np.where((highest_high_14 - lowest_low_14) != 0, 
                    100 * np.log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / np.log10(14), 
                    50)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price relative to KAMA
        above_kama = price_close > kama[i]
        below_kama = price_close < kama[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Regime filter: Choppiness Index low (trending market)
        trending_market = chop[i] < 38.2
        
        # Weekly trend filter
        weekly_uptrend = price_close > ema_20_1w_aligned[i]
        weekly_downtrend = price_close < ema_20_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Above KAMA, not overbought, trending market, weekly uptrend, volume spike
        long_signal = (above_kama and rsi_not_overbought and trending_market and 
                      weekly_uptrend and vol_confirmed)
        
        # Short: Below KAMA, not oversold, trending market, weekly downtrend, volume spike
        short_signal = (below_kama and rsi_not_oversold and trending_market and 
                       weekly_downtrend and vol_confirmed)
        
        # Exit when price crosses KAMA in opposite direction or RSI reaches extreme
        exit_long = position == 1 and (price_close < kama[i] or rsi[i] > 75)
        exit_short = position == -1 and (price_close > kama[i] or rsi[i] < 25)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals