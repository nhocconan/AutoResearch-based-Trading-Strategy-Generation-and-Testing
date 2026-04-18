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
    
    # Get 1D data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily data
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 12h timeframe
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 12h EMA34 for trend filter
    close_series = pd.Series(close)
    ema34_12h = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate RSI(14) on 12h
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
    
    start_idx = max(34, 200, 20)  # need EMA34, EMA200, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_12h[i]) or np.isnan(ema34_12h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long entry: price above daily EMA200, EMA34 trending up, RSI not overbought, with volume
            if (close[i] > ema200_12h[i] and 
                ema34_12h[i] > ema34_12h[i-1] and 
                rsi[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price below daily EMA200, EMA34 trending down, RSI not oversold, with volume
            elif (close[i] < ema200_12h[i] and 
                  ema34_12h[i] < ema34_12h[i-1] and 
                  rsi[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below EMA34 or RSI overbought
            if close[i] < ema34_12h[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA34 or RSI oversold
            if close[i] > ema34_12h[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA200_EMA34_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0