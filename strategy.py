#!/usr/bin/env python3
name = "6h_RSI_Contrarian_1dTrend_Filter"
timeframe = "6h"
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
    
    # Load 1D data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1D for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14) on 6H
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First values
    if len(gain) >= 14:
        avg_gain[13] = np.nanmean(gain[1:14])
        avg_loss[13] = np.nanmean(loss[1:14])
        
        # Wilder's smoothing
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close, np.nan)
    valid = ~np.isnan(avg_loss) & (avg_loss != 0)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    
    rsi = np.full_like(close, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1D EMA50 to 6H timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient data
        if (np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1D EMA50
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: RSI oversold (<30) in uptrend (price > 1D EMA50)
            if rsi[i] < 30 and price_above_ema50:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) in downtrend (price < 1D EMA50)
            elif rsi[i] > 70 and price_below_ema50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend reversal
            if rsi[i] > 70 or not price_above_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend reversal
            if rsi[i] < 30 or not price_below_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals