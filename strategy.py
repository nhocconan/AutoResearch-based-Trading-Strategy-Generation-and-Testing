#!/usr/bin/env python3
# 4h_donchian_1d_trend_volume_v1
# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and 1d EMA trend filter.
# In bull markets: price breaks above 1d Donchian upper + 1d EMA50 up + volume spike → long
# In bear markets: price breaks below 1d Donchian lower + 1d EMA50 down + volume spike → short
# Donchian channels provide clear structure, EMA50 filters trend direction, volume confirms momentum.
# Discrete sizing (±0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Donchian and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian Channel (20-period)
    donchian_period = 20
    upper_1d = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_1d = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = pd.Series(close_1d).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Volume confirmation: current 4h volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d EMA50 (trend reversal)
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d EMA50 (trend reversal)
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above 1d Donchian upper + price above 1d EMA50
                if close[i] > upper_1d_aligned[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 1d Donchian lower + price below 1d EMA50
                elif close[i] < lower_1d_aligned[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals