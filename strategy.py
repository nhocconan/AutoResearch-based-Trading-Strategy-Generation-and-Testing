#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ConnorsRSI_Donchian20_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi3 = 100 - (100 / (1 + rs))
    rsi3 = rsi3.fillna(50).values
    
    # RSI Streak (2)
    up_days = np.zeros(n)
    down_days = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_days[i] = up_days[i-1] + 1
            down_days[i] = 0
        elif close[i] < close[i-1]:
            down_days[i] = down_days[i-1] + 1
            up_days[i] = 0
        else:
            up_days[i] = up_days[i-1]
            down_days[i] = down_days[i-1]
    rsi_up = pd.Series(up_days).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rsi_down = pd.Series(down_days).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rsi_streak = 100 - (100 / (1 + rsi_up / rsi_down.replace(0, np.nan)))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Percent Rank (100)
    def rolling_percent_rank(arr, window=100):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            rank = np.sum(window_data < arr[i]) / (window-1) * 100
            result[i] = rank
        return result
    percent_rank = rolling_percent_rank(close, 100)
    
    # Connors RSI
    crsi = (rsi3 + rsi_streak + percent_rank) / 3
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 6h volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Donchian(20) breakout levels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20, 20, 100)  # EMA34, Donchian20, VolMA20, PercentRank100
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(crsi[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema34_1d_aligned[i]
        crsi_val = crsi[i]
        dh = highest_high[i]
        dl = lowest_low[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: CRSI < 15, price > Donchian high, above trend, volume filter
            if crsi_val < 15 and close[i] > dh and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: CRSI > 85, price < Donchian low, below trend, volume filter
            elif crsi_val > 85 and close[i] < dl and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CRSI > 70 or close below Donchian low
            if crsi_val > 70 or close[i] < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CRSI < 30 or close above Donchian high
            if crsi_val < 30 or close[i] > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals