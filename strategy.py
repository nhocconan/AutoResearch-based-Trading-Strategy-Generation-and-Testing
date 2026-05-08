#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w RSI bias and 1d volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly RSI > 50 AND daily volume > 20-day average.
# Short when price breaks below Donchian(20) low AND weekly RSI < 50 AND daily volume > 20-day average.
# Exit when price crosses back inside the Donchian channel.
# Uses 12h timeframe as specified, with 1w RSI for longer-term bias and 1d volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_Donchian_20_1wRSI_1dVolume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for RSI (trend bias)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 14:
        return np.zeros(n)
    
    # Daily data for volume confirmation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Donchian(20) on 12h data
    donchian_period = 20
    upper_dc = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Weekly RSI(14) for trend bias
    close_w = df_w['close'].values
    delta = np.diff(close_w, prepend=close_w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_w = 100 - (100 / (1 + rs))
    rsi_w = np.where(np.isnan(rsi_w), 50, rsi_w)  # Default to neutral if undefined
    
    # Align weekly RSI to 12h timeframe
    rsi_w_aligned = align_htf_to_ltf(prices, df_w, rsi_w)
    
    # Daily volume filter: current volume > 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > vol_ma20_d
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(rsi_w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, weekly RSI bullish, volume confirmation
            long_cond = (close[i] > upper_dc[i]) and (rsi_w_aligned[i] > 50) and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, weekly RSI bearish, volume confirmation
            short_cond = (close[i] < lower_dc[i]) and (rsi_w_aligned[i] < 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals