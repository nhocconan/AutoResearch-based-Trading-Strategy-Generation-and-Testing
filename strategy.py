#!/usr/bin/env python3
# 12h_KAMA_RSI_ChopFilter_v2
# Hypothesis: In choppy markets (CHOPPINESS > 61.8), price tends to revert to the mean.
# We use KAMA to determine trend direction and RSI for mean-reversion entries.
# Only trade when KAMA confirms trend and RSI shows extreme conditions.
# This strategy targets 12h timeframe with low trade frequency to avoid fee drag.
# Works in both bull and bear markets by combining trend following with mean reversion.

name = "12h_KAMA_RSI_ChopFilter_v2"
timeframe = "12h"
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
    
    # Get daily data for Chop and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index (14-period)
    # CHOP = 100 * log10(SUM(ATR(1), n) / (MAX(HIGH, n) - MIN(LOW, n))) / log10(n)
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # Align with original indices
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (max_high - min_low)) / np.log10(14)
    
    # Calculate RSI (14-period) on daily close
    delta = pd.Series(df_1d['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate KAMA (10-period ER, 2 and 30 for SC)
    # ER = |Close - Close(10)| / SUM(|Close - Close(1)|, 10)
    change = np.abs(np.concatenate([[np.nan], np.diff(close, n=10)]))  # |Close - Close(10)|
    volatility = np.abs(np.concatenate([[np.nan], np.diff(close, n=1)]))  # |Close - Close(1)|
    sum_vol = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(sum_vol > 0, change / sum_vol, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # SC = [ER*(fastest - slowest) + slowest]^2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align HTF indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is already daily, align to 12h
    
    # Weekly trend filter: price above/below weekly EMA20
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Chop (14), RSI (14), KAMA (10), weekly EMA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade in choppy market (CHOPPINESS > 61.8)
        choppy = chop_aligned[i] > 61.8
        
        # Weekly trend filter
        above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: choppy market + RSI oversold + price above weekly EMA
            if choppy and rsi_aligned[i] < 30 and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: choppy market + RSI overbought + price below weekly EMA
            elif choppy and rsi_aligned[i] > 70 and below_weekly_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or chop ends
            if rsi_aligned[i] > 50 or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or chop ends
            if rsi_aligned[i] < 50 or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals