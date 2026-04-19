#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_MeanReversion_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d RSI (14-period) with Wilder's smoothing
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, np.nan)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    rsi_len = 14
    avg_up = np.full_like(close_1d, np.nan)
    avg_down = np.full_like(close_1d, np.nan)
    
    # Initial simple average
    if len(up) >= rsi_len:
        avg_up[rsi_len-1] = np.nanmean(up[1:rsi_len+1])
        avg_down[rsi_len-1] = np.nanmean(down[1:rsi_len+1])
        
        # Wilder's smoothing
        for i in range(rsi_len, len(up)):
            avg_up[i] = (avg_up[i-1] * (rsi_len-1) + up[i]) / rsi_len
            avg_down[i] = (avg_down[i-1] * (rsi_len-1) + down[i]) / rsi_len
    
    rs = np.divide(avg_up, avg_down, out=np.full_like(avg_up, np.nan), where=avg_down!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12h price moving averages for trend filter
    close_s = pd.Series(close)
    ma_fast = close_s.ewm(span=25, adjust=False, min_periods=25).mean().values
    ma_slow = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 25)
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        rsi = rsi_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Volume filter: current volume > 1.3x 20-day average
        volume_ok = vol > 1.3 * vol_ma
        
        # Trend filter: bullish when fast MA > slow MA
        bullish_trend = ma_fast[i] > ma_slow[i]
        bearish_trend = ma_fast[i] < ma_slow[i]
        
        if position == 0:
            # Long: RSI oversold (<30) in bullish trend with volume confirmation
            if rsi < 30 and bullish_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in bearish trend with volume confirmation
            elif rsi > 70 and bearish_trend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (50) or trend changes to bearish
            if rsi >= 50 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (50) or trend changes to bullish
            if rsi <= 50 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals