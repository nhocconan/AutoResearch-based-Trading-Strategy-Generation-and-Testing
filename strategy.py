#!/usr/bin/env python3
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
    
    # Get 1d data for ATR and price channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period ATR on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(low_1d[1:] - close_1d[:-1], np.abs(low_1d[1:] - high_1d[:-1]))
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr = np.zeros_like(high_1d)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 19 + tr[i]) / 20 if i >= 20 else np.nan
    
    # Calculate 20-period Donchian channel on 1d
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align ATR and Donchian to 4h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi[i])):
            continue
        
        # Breakout with volume and RSI filter
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        vol_filter = volume[i] > np.mean(volume[max(0, i-19):i+1])
        rsi_filter = (rsi[i] > 30) & (rsi[i] < 70)
        
        long_entry = breakout_long and vol_filter and rsi_filter
        short_entry = breakout_short and vol_filter and rsi_filter
        
        # Exit: opposite breakout or ATR stop
        if position == 1:
            exit_condition = (breakout_short or 
                            close[i] < donchian_high_aligned[i] - 1.5 * atr_aligned[i])
        elif position == -1:
            exit_condition = (breakout_long or 
                            close[i] > donchian_low_aligned[i] + 1.5 * atr_aligned[i])
        else:
            exit_condition = False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_condition:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = position_size * position
    
    return signals

name = "4h_1d_donchian_atr_rsi_filter"
timeframe = "4h"
leverage = 1.0