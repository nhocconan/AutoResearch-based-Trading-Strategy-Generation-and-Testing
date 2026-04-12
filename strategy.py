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
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for price position
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period high and low for Donchian channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high (20-period high)
    donchian_high = np.full(len(high_1d), np.nan)
    for i in range(19, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
    
    # Donchian low (20-period low)
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(19, len(low_1d)):
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 4h ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-14:i+1])
    
    # 20-period ATR average
    atr_ma = np.full(n, np.nan)
    for i in range(34, n):
        atr_ma[i] = np.nanmean(atr[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period ATR mean
        vol_filter = atr[i] > atr_ma[i] * 0.5
        
        # Entry conditions: Breakout of Donchian channel with weekly RSI filter
        long_breakout = close[i] > donchian_high_4h[i]
        short_breakout = close[i] < donchian_low_4h[i]
        
        # RSI filter: Avoid overbought/oversold extremes
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        long_entry = long_breakout and vol_filter and rsi_not_overbought
        short_entry = short_breakout and vol_filter and rsi_not_oversold
        
        # Exit conditions: Return to midpoint of Donchian channel
        donchian_mid = (donchian_high_4h[i] + donchian_low_4h[i]) / 2
        long_exit = close[i] < donchian_mid
        short_exit = close[i] > donchian_mid
        
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

name = "4h_1w_donchian_breakout_rsi_filter_v1"
timeframe = "4h"
leverage = 1.0