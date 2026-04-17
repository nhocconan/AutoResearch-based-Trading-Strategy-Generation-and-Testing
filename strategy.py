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
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for entry timing
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA200 for long-term trend
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Weekly trend: price above/below EMA200
    weekly_trend_up = df_1w['close'].values > ema_200_1w
    weekly_trend_down = df_1w['close'].values < ema_200_1w
    
    # Align weekly trend to 6h (waits for weekly bar to close)
    weekly_trend_up_6h = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_6h = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Daily Donchian(20) for breakout
    highest_20_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h
    highest_20_6h = align_htf_to_ltf(prices, df_1d, highest_20_1d)
    lowest_20_6h = align_htf_to_ltf(prices, df_1d, lowest_20_1d)
    
    # Volume confirmation: 20-period volume MA on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20_6h[i]) or np.isnan(lowest_20_6h[i]) or
            np.isnan(weekly_trend_up_6h[i]) or np.isnan(weekly_trend_down_6h[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: weekly uptrend + price breaks above daily Donchian high + volume spike
            if (weekly_trend_up_6h[i] > 0.5 and 
                price > highest_20_6h[i] and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below daily Donchian low + volume spike
            elif (weekly_trend_down_6h[i] > 0.5 and 
                  price < lowest_20_6h[i] and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly EMA200 or volume drops below average
            if (price < ema_200_1w[-1] if len(ema_200_1w) > 0 else False) or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly EMA200 or volume drops below average
            if (price > ema_200_1w[-1] if len(ema_200_1w) > 0 else False) or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyTrend_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0