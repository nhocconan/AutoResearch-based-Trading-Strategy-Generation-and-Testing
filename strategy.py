#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_ChopFilter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) to detect trend direction on 12h timeframe, combined with RSI momentum and Choppiness Index regime filter to avoid whipsaws in sideways markets. Designed for low trade frequency (~15-30 trades/year) to minimize fee drag while capturing sustained trends in both bull and bear markets.
"""

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
    
    # Get 1d data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    atr_14 = []
    for i in range(len(df_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            tr = np.max([
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            ])
            atr_14.append(tr)
    
    # Smooth ATR with Wilder's smoothing (equivalent to RMA)
    atr_smoothed = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 14:
            atr_smoothed[i] = np.nan
        elif i == 14:
            atr_smoothed[i] = np.nanmean(atr_14[1:15])
        else:
            atr_smoothed[i] = (atr_smoothed[i-1] * 13 + atr_14[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Chop = log10(sum(atr14) / (hh14 - ll14)) * 100 / log10(14)
    chop_raw = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if (i < 14) or np.isnan(atr_smoothed[i]) or np.isnan(hh_14[i]) or np.isnan(ll_14[i]) or (hh_14[i] - ll_14[i]) == 0:
            chop_raw[i] = np.nan
        else:
            chop_raw[i] = np.log10(atr_smoothed[i] * 14) / np.log10(hh_14[i] - ll_14[i]) * 100
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    changes = np.diff(df_12h['close'])
    abs_changes = np.abs(changes)
    
    er = np.full(len(df_12h), np.nan)
    for i in range(9, len(df_12h)):  # ER needs 10 periods (current + 9 prior)
        net_change = abs(df_12h['close'].iloc[i] - df_12h['close'].iloc[i-9])
        sum_abs = np.sum(abs_changes[i-9:i]) if i >= 9 else np.nan
        if sum_abs > 0:
            er[i] = net_change / sum_abs
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(len(df_12h), np.nan)
    kama[9] = df_12h['close'].iloc[9]  # Start after ER period
    for i in range(10, len(df_12h)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (df_12h['close'].iloc[i] - kama[i-1])
    
    # Get 12h data for RSI (14)
    rsi_period = 14
    delta = np.diff(df_12h['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full(len(df_12h), np.nan)
    avg_loss = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        if i < rsi_period:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == rsi_period:
            avg_gain[i] = np.mean(gain[1:rsi_period+1])
            avg_loss[i] = np.mean(loss[1:rsi_period+1])
        else:
            if avg_gain[i-1] == 0:
                avg_gain[i] = (gain[i]) / rsi_period
            else:
                avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            if avg_loss[i-1] == 0:
                avg_loss[i] = (loss[i]) / rsi_period
            else:
                avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rsi = np.full(len(df_12h), np.nan)
    for i in range(rsi_period, len(df_12h)):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Align all higher timeframe data to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 61.8 = ranging (avoid trend following)
        # Chop < 38.2 = trending (favor trend following)
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        # KAMA trend: price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI momentum: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Entry conditions
        # Long: trending market + price above KAMA + RSI not overbought
        long_entry = is_trending and price_above_kama and rsi_not_overbought
        
        # Short: trending market + price below KAMA + RSI not oversold
        short_entry = is_trending and price_below_kama and rsi_not_oversold
        
        # Exit conditions: opposite signal or ranging market
        long_exit = (not price_above_kama) or is_ranging
        short_exit = (not price_below_kama) or is_ranging
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0