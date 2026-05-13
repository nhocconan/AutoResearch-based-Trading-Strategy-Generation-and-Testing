#!/usr/bin/env python3
name = "4h_RSI_Regime_Breakout"
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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Choppiness regime: avoid trending markets (CHOP < 38.2)
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    sum_tr = np.zeros(n)
    for i in range(atr_period, n):
        sum_tr[i] = np.sum(tr[i-atr_period+1:i+1])
    highest = np.maximum.accumulate(high)
    lowest = np.minimum.accumulate(low)
    range_max_min = highest - lowest
    chop = np.where(range_max_min != 0, 100 * np.log10(sum_tr / range_max_min) / np.log10(atr_period), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(rsi[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        chop_filter = chop[i] > 50  # favor ranging markets
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # LONG: RSI oversold + 1d uptrend + volume + chop
            if rsi_oversold and close[i] > ema34_1d_aligned[i] and vol_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought + 1d downtrend + volume + chop
            elif rsi_overbought and close[i] < ema34_1d_aligned[i] and vol_filter and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 or trend breaks
            if rsi[i] > 50 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 or trend breaks
            if rsi[i] < 50 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals