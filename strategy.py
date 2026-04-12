#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter
# Long when price breaks above 20-day high and weekly price > 40-week EMA (bullish trend)
# Short when price breaks below 20-day low and weekly price < 40-week EMA (bearish trend)
# Exit when price crosses 20-day EMA (trend reversal signal)
# Target: 20-40 trades/year on daily timeframe with strong trend filter to avoid whipsaws
# Works in bull markets via breakouts and bear markets via short breakdowns with trend confirmation

name = "1d_20d_donchian_weekly_ema_trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels (breakout levels)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    ema_20 = pd.Series(df_1d['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align daily indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Get weekly data for trend filter (40-week EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 40-week EMA for trend filter
    weekly_ema_40 = pd.Series(df_1w['close']).ewm(span=40, min_periods=40, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_40)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above 20-day high AND weekly trend is bullish
        if (close[i] > donchian_high_aligned[i] and 
            close[i] > weekly_ema_aligned[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below 20-day low AND weekly trend is bearish
        elif (close[i] < donchian_low_aligned[i] and 
              close[i] < weekly_ema_aligned[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses 20-day EMA (trend reversal)
        elif position == 1 and close[i] < ema_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals