#!/usr/bin/env python3
"""
1d_donchian_20_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, enter long when price breaks above 20-day Donchian high with weekly trend confirmation and volume surge; enter short when price breaks below 20-day Donchian low with weekly trend confirmation and volume surge. Exit on opposite breakout or when price returns to 20-day moving average. Uses weekly trend filter to avoid counter-trend trades in strong trends. Targets 7-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility filter and exit
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Donchian channels (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels: based on past 20 days' high/low
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (shifted by 1 day to avoid look-ahead)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Align to 1d timeframe (shifted by 1 week to avoid look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or atr[i] <= 0 or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on short signal (price breaks below 20-day low with volume)
            if close[i] < low_20_aligned[i] and vol_confirm:
                exit_long = True
            # Exit when price returns to 20-day moving average (mean reversion)
            elif abs(close[i] - high_20_aligned[i]) < 1.0 * atr[i]:  # Using high_20 as proxy for mid-point
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on long signal (price breaks above 20-day high with volume)
            if close[i] > high_20_aligned[i] and vol_confirm:
                exit_short = True
            # Exit when price returns to 20-day moving average (mean reversion)
            elif abs(close[i] - low_20_aligned[i]) < 1.0 * atr[i]:  # Using low_20 as proxy for mid-point
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with volume confirmation AND weekly uptrend
            long_entry = (close[i] > high_20_aligned[i] and 
                         vol_confirm and 
                         close[i] > ema_50_aligned[i])
            
            # Short entry: price breaks below 20-day low with volume confirmation AND weekly downtrend
            short_entry = (close[i] < low_20_aligned[i] and 
                          vol_confirm and 
                          close[i] < ema_50_aligned[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals