# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week Donchian channel breakout with 1-day volatility filter
# and volume confirmation. Weekly Donchian provides strong trend-following signals that
# work in both bull and bear markets by capturing major breakouts. 1-day ATR filter
# avoids false signals during low volatility periods. Volume confirmation ensures
# institutional participation. Designed for low trade frequency (<30/year) to minimize
# fee drag while capturing significant moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian Channel on 1w data (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_1w).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_1w).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Load 1d data ONCE for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Load 1d data for volatility reference (20-period average ATR)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1w timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Volume confirmation: 2.0x 50-period average volume
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donchian_period, 50)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(atr_ma_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 0.5 * 20-day average ATR
        # Avoids trading during extremely low volatility periods
        vol_filter = atr_aligned[i] > 0.5 * atr_ma_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian channel breakouts
            # Long: price breaks above upper Donchian channel
            if (close[i] > upper_channel_aligned[i] and 
                vol_filter and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel
            elif (close[i] < lower_channel_aligned[i] and 
                  vol_filter and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to the midpoint of the Donchian channel
            midpoint = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if close[i] <= midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to the midpoint of the Donchian channel
            midpoint = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if close[i] >= midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wDonchian_Breakout_VolVolFilter_v1"
timeframe = "12h"
leverage = 1.0