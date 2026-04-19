#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Works in bull: breaks above upper band with volume
# Works in bear: breaks below lower band with volume (short)
# Uses trend filter to avoid counter-trend trades, reducing whipsaw
# Target: 20-40 trades/year to minimize fee drag
name = "4h_Donchian20_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: 08-20 UTC (active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to be valid
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema34_1d_aligned[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume + uptrend
            if price > high_roll[i] and volume_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume + downtrend
            elif price < low_roll[i] and volume_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below lower Donchian (reversal)
            if price < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above upper Donchian (reversal)
            if price > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals