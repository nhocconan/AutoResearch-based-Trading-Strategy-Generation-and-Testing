#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 20-period Donchian channel (trend structure)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian upper and lower bands (20-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_high = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian bands to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Get daily data for 14-period RSI (momentum filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI on daily data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # need Donchian, RSI, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(rsi_1d_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high, RSI not overbought, with volume
            if (close[i] > donchian_high_12h[i] and 
                rsi_1d_12h[i] < 60 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low, RSI not oversold, with volume
            elif (close[i] < donchian_low_12h[i] and 
                  rsi_1d_12h[i] > 40 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low or RSI overbought
            if close[i] < donchian_low_12h[i] or rsi_1d_12h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high or RSI oversold
            if close[i] > donchian_high_12h[i] or rsi_1d_12h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchian_Breakout_RSI_Volume"
timeframe = "12h"
leverage = 1.0