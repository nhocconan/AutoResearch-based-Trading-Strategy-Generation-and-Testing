#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 200-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(200) for trend
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 4-hour data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20) channels
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h RSI(14)
    delta = np.diff(close_4h)
    delta = np.insert(delta, 0, np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_4h = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 50)
    
    # Calculate 4h volume moving average
    vol_s_4h = pd.Series(volume_4h)
    vol_ma_20_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(high_20_4h_aligned[i]) or np.isnan(low_20_4h_aligned[i]) or 
            np.isnan(rsi_14_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period 4h volume MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Trend filter: price above/below daily EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_14_4h_aligned[i] < 70
        rsi_not_oversold = rsi_14_4h_aligned[i] > 30
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_4h_aligned[i]
        short_breakout = close[i] < low_20_4h_aligned[i]
        
        # Entry conditions: breakout in trend direction + volume + RSI filter
        long_entry = long_breakout and uptrend and vol_filter and rsi_not_overbought
        short_entry = short_breakout and downtrend and vol_filter and rsi_not_oversold
        
        # Exit conditions: opposite breakout or RSI extreme
        long_exit = (close[i] < low_20_4h_aligned[i]) or (rsi_14_4h_aligned[i] > 80)
        short_exit = (close[i] > high_20_4h_aligned[i]) or (rsi_14_4h_aligned[i] < 20)
        
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

name = "4h_1d_ema200_rsi_vol_filter_v1"
timeframe = "4h"
leverage = 1.0