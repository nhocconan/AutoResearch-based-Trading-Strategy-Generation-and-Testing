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
    
    # Get daily data for Donchian channels and RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily timeframe
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    
    # Calculate 14-period RSI on daily timeframe
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(gain) >= 14:
        avg_gain[14] = np.mean(gain[:14])
        avg_loss[14] = np.mean(loss[:14])
        for i in range(15, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = np.full_like(close_1d, 100.0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to 4h timeframe
    upper_4h = align_htf_to_ltf(prices, df_1d, upper)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average (moderate threshold)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(rsi_4h[i]) or np.isnan(ema_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above weekly EMA (bullish bias)
        bullish_bias = close[i] > ema_1w_4h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and RSI not overbought
            if close[i] > upper_4h[i] and vol_confirm and rsi_4h[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and RSI not oversold
            elif close[i] < lower_4h[i] and vol_confirm and rsi_4h[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR RSI overbought
            if close[i] < lower_4h[i] or rsi_4h[i] > 70:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR RSI oversold
            if close[i] > upper_4h[i] or rsi_4h[i] < 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_RSI_Volume_Filter"
timeframe = "4h"
leverage = 1.0