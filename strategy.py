#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_Chop"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Pad ER to match close length
    er_padded = np.concatenate([np.full(er_length-1, np.nan), er])
    
    # Smoothing constants
    sc = (er_padded * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_length-1] = close[er_length-1]  # seed
    for i in range(er_length, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 4h (already on 4h, but ensure no look-ahead)
    kama_aligned = kama  # already calculated on close prices
    
    # RSI(14) on close
    def rsi(series, period=14):
        delta = np.diff(series, n=1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(series)
        avg_loss = np.zeros_like(series)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        rsi_vals[:period] = np.nan
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Choppiness Index (14) - using 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with high_1d
    
    # ATR(14)
    atr = np.zeros_like(tr)
    atr[14] = np.nanmean(tr[1:15])  # seed
    for i in range(15, len(tr)):
        if not np.isnan(tr[i]):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.zeros_like(tr)
    for i in range(14, len(tr)):
        sum_atr_14[i] = np.nansum(tr[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(tr)
    max_hh = np.maximum.accumulate(high_1d)
    min_ll = np.minimum.accumulate(low_1d)
    range_14 = max_hh - min_ll
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop[:14] = np.nan
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter (20-period average)
    vol_ma = np.zeros_like(volume)
    vol_ma[20:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:20] = np.nan
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20, 14)  # KAMA seed, vol MA, chop period
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + chop < 61.8 (trending) + volume
            if kama_aligned[i] > kama_aligned[i-1] and rsi_vals[i] > 50 and chop_aligned[i] < 61.8 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + chop < 61.8 (trending) + volume
            elif kama_aligned[i] < kama_aligned[i-1] and rsi_vals[i] < 50 and chop_aligned[i] < 61.8 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or RSI < 40 or chop > 61.8 (choppy)
            if kama_aligned[i] < kama_aligned[i-1] or rsi_vals[i] < 40 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or RSI > 60 or chop > 61.8 (choppy)
            if kama_aligned[i] > kama_aligned[i-1] or rsi_vals[i] > 60 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals