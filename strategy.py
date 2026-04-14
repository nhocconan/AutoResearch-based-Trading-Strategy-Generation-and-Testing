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
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 6h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 6h timeframe
    atr_6h_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    # Calculate 6h price position relative to 1d range
    # Use previous day's range for today's context
    high_1d_prev = np.concatenate([[high_1d[0]], high_1d[:-1]])  # yesterday's high
    low_1d_prev = np.concatenate([[low_1d[0]], low_1d[:-1]])    # yesterday's low
    range_1d_prev = high_1d_prev - low_1d_prev
    
    # Normalize current price to yesterday's range (0 = low, 1 = high)
    price_pos_1d = (close - low_1d_prev) / range_1d_prev
    price_pos_1d = np.clip(price_pos_1d, 0, 1)  # clamp to [0,1]
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_aligned[i]) or
            np.isnan(atr_6h_aligned[i]) or
            np.isnan(volume_ma[i]) or
            np.isnan(price_pos_1d[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h_aligned[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Avoid extreme overbought/oversold conditions
        if rsi_14_aligned[i] > 80 or rsi_14_aligned[i] < 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price in lower third of yesterday's range with bullish momentum
            if price_pos_1d[i] < 0.33 and rsi_14_aligned[i] > 50:
                position = 1
                signals[i] = position_size
            # Short: Price in upper third of yesterday's range with bearish momentum
            elif price_pos_1d[i] > 0.66 and rsi_14_aligned[i] < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price reaches upper third or momentum fades
            if price_pos_1d[i] >= 0.66 or rsi_14_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price reaches lower third or momentum fades
            if price_pos_1d[i] <= 0.33 or rsi_14_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_RangePosition_RSI_Filter"
timeframe = "6h"
leverage = 1.0