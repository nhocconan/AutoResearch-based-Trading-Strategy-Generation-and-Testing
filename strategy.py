#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    high_20d = np.full(len(high_1d), np.nan)
    low_20d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        high_20d[i] = np.max(high_1d[i-20:i])
        low_20d[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian channels to 12h timeframe
    high_20d_12h = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_12h = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 34-week EMA for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to 12h timeframe
    ema34_1w_12h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # need Donchian, EMA34, RSI, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20d_12h[i]) or np.isnan(low_20d_12h[i]) or 
            np.isnan(ema34_1w_12h[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above 20-day high, weekly trend up, RSI not overbought, with volume
            if (close[i] > high_20d_12h[i] and 
                ema34_1w_12h[i] > ema34_1w_12h[i-1] and 
                rsi[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low, weekly trend down, RSI not oversold, with volume
            elif (close[i] < low_20d_12h[i] and 
                  ema34_1w_12h[i] < ema34_1w_12h[i-1] and 
                  rsi[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or RSI overbought
            if close[i] < low_20d_12h[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or RSI oversold
            if close[i] > high_20d_12h[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyEMA34_RSI_Volume"
timeframe = "12h"
leverage = 1.0