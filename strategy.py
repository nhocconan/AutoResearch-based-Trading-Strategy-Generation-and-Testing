#!/usr/bin/env python3
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
    
    # Load daily data for 12h strategy
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ADX for trend strength (daily)
    if len(high_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - high_1d[i-1]),
                   abs(low_1d[i] - low_1d[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    atr_14 = np.full_like(high_1d, np.nan)
    plus_dm_14 = np.full_like(high_1d, np.nan)
    minus_dm_14 = np.full_like(high_1d, np.nan)
    
    if len(high_1d) >= 14:
        atr_14[13] = np.mean(tr[1:14])
        plus_dm_14[13] = np.mean(plus_dm[1:14])
        minus_dm_14[13] = np.mean(minus_dm[1:14])
        for i in range(14, len(high_1d)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # DI and DX
    plus_di_14 = np.full_like(high_1d, np.nan)
    minus_di_14 = np.full_like(high_1d, np.nan)
    dx_14 = np.full_like(high_1d, np.nan)
    
    for i in range(13, len(high_1d)):
        if atr_14[i] > 0:
            plus_di_14[i] = 100 * plus_dm_14[i] / atr_14[i]
            minus_di_14[i] = 100 * minus_dm_14[i] / atr_14[i]
            if plus_di_14[i] + minus_di_14[i] > 0:
                dx_14[i] = 100 * abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])
    
    # ADX (smoothed DX)
    adx_14 = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 27:  # 14 + 13 for smoothing
        adx_14[26] = np.mean(dx_14[14:27])
        for i in range(27, len(high_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    # Align ADX to 12h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 20-period EMA for trend direction (daily)
    if len(close_1d) < 20:
        return np.zeros(n)
    
    ema20_1d = np.full_like(close_1d, np.nan)
    alpha = 2 / (20 + 1)
    ema20_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema20_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema20_1d[i-1]
    
    # Align EMA to 12h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 14-period RSI for momentum (daily)
    if len(close_1d) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1d, np.nan)
    rsi_14 = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_14[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_14[i] = 100 if avg_gain[i] > 0 else 0
    
    # Align RSI to 12h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 20-period ATR for volatility (daily)
    atr_20 = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 20:
        atr_20[19] = np.mean(tr[1:20])
        for i in range(20, len(high_1d)):
            atr_20[i] = (atr_20[i-1] * 19 + tr[i]) / 20
    
    # Align ATR to 12h timeframe
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size: 25% of capital
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(atr_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current period volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: ADX > 25 (trending) + price above EMA20 + RSI > 50 + volume surge
            if (adx_14_aligned[i] > 25 and 
                close[i] > ema20_1d_aligned[i] and
                rsi_14_aligned[i] > 50 and
                volume_ratio > 3.0):
                position = 1
                signals[i] = position_size
            # Short: ADX > 25 (trending) + price below EMA20 + RSI < 50 + volume surge
            elif (adx_14_aligned[i] > 25 and 
                  close[i] < ema20_1d_aligned[i] and
                  rsi_14_aligned[i] < 50 and
                  volume_ratio > 3.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ADX drops below 20 (trend weakening) or price crosses below EMA20
            if (adx_14_aligned[i] < 20 or
                close[i] < ema20_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: ADX drops below 20 (trend weakening) or price crosses above EMA20
            if (adx_14_aligned[i] < 20 or
                close[i] > ema20_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ADX_EMA_RSI_Volume_v1"
timeframe = "12h"
leverage = 1.0