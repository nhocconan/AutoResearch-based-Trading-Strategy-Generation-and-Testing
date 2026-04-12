#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h KAMA trend + RSI(14) mean reversion + 1d chop regime filter
    # KAMA adapts to market noise - trend following in trending markets, flat in choppy
    # RSI < 30 = oversold (long), RSI > 70 = overbought (short) only in KAMA trend direction
    # 1d Choppiness Index > 61.8 = choppy regime (avoid trend trades), < 38.2 = trending regime
    # Volume > 1.3x 20-period MA confirms momentum
    # Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR(14) for Choppiness Index
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with 1d indices
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate 1d Choppiness Index: CI = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    sum_atr_14 = np.full(len(df_1d), np.nan)
    max_high_14 = np.full(len(df_1d), np.nan)
    min_low_14 = np.full(len(df_1d), np.nan)
    
    for i in range(14, len(df_1d)):
        sum_atr_14[i] = np.sum(atr_14_1d[i-13:i+1])
        max_high_14[i] = np.max(high_1d[i-13:i+1])
        min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if max_high_14[i] > min_low_14[i] and sum_atr_14[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_atr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0  # neutral
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate KAMA(10,2,30) on 12h
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change_10 = np.abs(np.subtract(close[10:], close[:-10]))
    volatility_10 = np.full(n, np.nan)
    
    for i in range(10, n):
        volatility_10[i] = np.sum(np.abs(np.subtract(close[i-9:i+1], close[i-10:i])))
    
    er = np.full(n, np.nan)
    for i in range(10, n):
        if volatility_10[i] > 0:
            er[i] = change_10[i-10] / volatility_10[i]
        else:
            er[i] = 0.0
    
    # Smoothing constants: fastest SC = 2/(2+1) = 0.6667, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100.0 if avg_gain[i] > 0 else 0.0
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # RSI mean reversion conditions
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        
        # Chop regime filter: only allow trend trades in trending regime (CHOP < 38.2)
        # In choppy regime (CHOP > 61.8), we avoid trades to prevent whipsaw
        chop = chop_1d_aligned[i]
        trending_regime = chop < 38.2
        choppy_regime = chop > 61.8
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = oversold and (vol_ratio[i] > 1.3) and uptrend and trending_regime
        short_entry = overbought and (vol_ratio[i] > 1.3) and downtrend and trending_regime
        
        # Exit conditions: RSI returns to midpoint (50)
        long_exit = rsi[i] > 50
        short_exit = rsi[i] < 50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_kama_rsi_chop_vol_v1"
timeframe = "12h"
leverage = 1.0