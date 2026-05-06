#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with weekly trend filter and volume confirmation
# Uses Donchian(20) breakout on 12h timeframe for entry signals
# Uses weekly EMA(200) to filter for long-term trend direction
# Volume confirmation (>1.3x 20-bar average) ensures participation
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: breakouts capture trends, weekly filter avoids counter-trend trades

name = "12h_Donchian20_WeeklyEMA200_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_12h) < 20 or len(df_weekly) < 200:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) channels on 12h timeframe
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA(200) trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema_200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for 12h timeframe (for stoploss)
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe (primary)
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    weekly_ema_200_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_200)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(weekly_ema_200_aligned[i]) or np.isnan(atr_12h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Donchian channel AND above weekly EMA200 AND volume confirmation
            if (close[i] > upper_channel_aligned[i] and close[i] > weekly_ema_200_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian channel AND below weekly EMA200 AND volume confirmation
            elif (close[i] < lower_channel_aligned[i] and close[i] < weekly_ema_200_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below the lower Donchian channel
            if close[i] < lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above the upper Donchian channel
            if close[i] > upper_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals