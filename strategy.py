#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend direction
    close_weekly = df_weekly['close'].values
    ema200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily timeframe (waits for weekly close)
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Calculate daily ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x average volume (30-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=30, min_periods=30).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 30)  # 200 for weekly EMA, 30 for volume avg
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_weekly_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume confirmation
            if (price > highest_high[i] and 
                price > ema200_weekly_aligned[i] and 
                vol > 1.8 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + weekly downtrend + volume confirmation
            elif (price < lowest_low[i] and 
                  price < ema200_weekly_aligned[i] and 
                  vol > 1.8 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes
            if price < lowest_low[i] or price < ema200_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes
            if price > highest_high[i] or price > ema200_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Trend_Volume_Filter"
timeframe = "1d"
leverage = 1.0