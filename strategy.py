#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price channel breakout with 1-day trend filter and volume confirmation.
# Uses daily Donchian channels (20-period) to identify breakouts, daily SMA200 for trend filter,
# and volume spike for confirmation. Designed for low trade frequency (<40/year) to minimize
# fee drag while capturing strong momentum moves in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on daily data
    upper_20d = np.full(len(high_1d), np.nan)
    lower_20d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_20d[i] = np.max(high_1d[i-20:i])
        lower_20d[i] = np.min(low_1d[i-20:i])
    
    # Align daily Donchian channels to 12h timeframe
    upper_20d_aligned = align_htf_to_ltf(prices, df_1d, upper_20d)
    lower_20d_aligned = align_htf_to_ltf(prices, df_1d, lower_20d)
    
    # Calculate SMA(200) on daily close for trend filter
    sma_200_1d = np.full(len(close_1d), np.nan)
    for i in range(200, len(close_1d)):
        sma_200_1d[i] = np.mean(close_1d[i-200:i])
    
    # Align daily SMA200 to 12h timeframe
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate volume moving average (20-period) on 12h data
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # need daily SMA200, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20d_aligned[i]) or np.isnan(lower_20d_aligned[i]) or 
            np.isnan(sma_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above daily SMA200 (uptrend) or below (downtrend)
        trend_up = close[i] > sma_200_1d_aligned[i]
        trend_down = close[i] < sma_200_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above 20-day high, with volume and trend filter
            if (close[i] > upper_20d_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low, with volume and trend filter
            elif (close[i] < lower_20d_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 20-day low or opposite signal
            if close[i] < lower_20d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day high or opposite signal
            if close[i] > upper_20d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyDonchian20_SMA200_Volume2x"
timeframe = "12h"
leverage = 1.0