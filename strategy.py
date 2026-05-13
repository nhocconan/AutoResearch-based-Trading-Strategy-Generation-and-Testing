#!/usr/bin/env python3
name = "4h_RSI_Divergence_With_Volume_Filter"
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
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:i+1])
            avg_loss[i] = np.mean(loss[1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily trend filter using EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.3 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # RSI divergence detection (bearish: price HH, RSI LH; bullish: price LL, RSI HL)
    bearish_div = np.zeros(n, dtype=bool)
    bullish_div = np.zeros(n, dtype=bool)
    lookback = 10
    
    for i in range(lookback, n):
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (high[i] == np.max(high[i-lookback:i+1]) and 
            rsi[i] < np.max(rsi[i-lookback:i])):
            # Check if this is a new high in price
            if high[i] > np.max(high[i-lookback:i]):
                bearish_div[i] = True
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (low[i] == np.min(low[i-lookback:i+1]) and 
            rsi[i] > np.min(rsi[i-lookback:i])):
            # Check if this is a new low in price
            if low[i] < np.min(low[i-lookback:i]):
                bullish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(rsi[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_condition = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Bullish RSI divergence with bullish trend and volume
            if bullish_div[i] and close[i] > ema34_1d_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish RSI divergence with bearish trend and volume
            elif bearish_div[i] and close[i] < ema34_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish RSI divergence or trend reversal
            if bearish_div[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish RSI divergence or trend reversal
            if bullish_div[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals