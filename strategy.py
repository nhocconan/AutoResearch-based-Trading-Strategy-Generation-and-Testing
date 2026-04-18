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
    
    # Get 1D data for Donchian(20) channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) on daily data: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper_1d = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 12h timeframe (will be available after 20-day period completes)
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Calculate 12-period RSI on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)  # need Donchian(20) and RSI(12)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper, RSI not overbought, with volume
            if (close[i] > donchian_upper_12h[i] and 
                rsi[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower, RSI not oversold, with volume
            elif (close[i] < donchian_lower_12h[i] and 
                  rsi[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or RSI overbought
            if close[i] < donchian_lower_12h[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or RSI oversold
            if close[i] > donchian_upper_12h[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_RSI12_VolumeFilter"
timeframe = "12h"
leverage = 1.0