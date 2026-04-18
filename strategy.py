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
    
    # Get 1D data for OHLC and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Align pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 4h EMA34 for trend filter
    close_series = pd.Series(close)
    ema34_4h = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume moving average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(ema34_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price touches S1 or S2 with volume and EMA34 trending up
            if ((abs(close[i] - s1_4h[i]) < 0.001 * close[i] or abs(close[i] - s2_4h[i]) < 0.001 * close[i]) and 
                ema34_4h[i] > ema34_4h[i-1] and 
                rsi[i] < 70 and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: price touches R1 or R2 with volume and EMA34 trending down
            elif ((abs(close[i] - r1_4h[i]) < 0.001 * close[i] or abs(close[i] - r2_4h[i]) < 0.001 * close[i]) and 
                  ema34_4h[i] < ema34_4h[i-1] and 
                  rsi[i] > 30 and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price touches R1 or R2 or RSI overbought
            if (abs(close[i] - r1_4h[i]) < 0.001 * close[i] or abs(close[i] - r2_4h[i]) < 0.001 * close[i] or 
                rsi[i] > 75):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches S1 or S2 or RSI oversold
            if (abs(close[i] - s1_4h[i]) < 0.001 * close[i] or abs(close[i] - s2_4h[i]) < 0.001 * close[i] or 
                rsi[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1S2_R1R2_Touch_EMA34_Volume"
timeframe = "4h"
leverage = 1.0