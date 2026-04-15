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
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-period volume moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Get aligned indicators
        ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)[i]
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        vol_ma_20_val = vol_ma_20[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(ema_20_aligned) or np.isnan(rsi_1d_aligned) or 
            np.isnan(vol_ma_20_val) or np.isnan(high_20[i]) or np.isnan(low_20[i])):
            continue
        
        # Volume confirmation (> 1.2x average)
        volume_confirm = volume[i] > 1.2 * vol_ma_20_val
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: Break above 20-period high + price above 1d EMA20 + RSI not overbought
                if high[i] > high_20[i] and close[i] > ema_20_aligned and rsi_1d_aligned < 70:
                    position = 1
                    signals[i] = position_size
                # Short: Break below 20-period low + price below 1d EMA20 + RSI not oversold
                elif low[i] < low_20[i] and close[i] < ema_20_aligned and rsi_1d_aligned > 30:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below 20-period low
            if low[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above 20-period high
            if high[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA20_RSI_Volume_v1"
timeframe = "4h"
leverage = 1.0