#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; mean reversion from extreme levels
# works well in ranging markets. 1d EMA34 ensures trades align with higher timeframe trend
# to avoid fighting the trend. Volume confirmation (1.8x 20-period EMA) filters weak signals.
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years)
# with discrete sizing (0.25). Works in bull markets by buying pullbacks in uptrends
# and in bear markets by selling rallies in downtrends.

name = "6h_WilliamsR_MeanReversion_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.8x 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + volume spike + price above 1d EMA34 (uptrend)
            if (williams_r_1d_aligned[i] < -80 and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + volume spike + price below 1d EMA34 (downtrend)
            elif (williams_r_1d_aligned[i] > -20 and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 OR price below 1d EMA34 (trend change)
            if williams_r_1d_aligned[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 OR price above 1d EMA34 (trend change)
            if williams_r_1d_aligned[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals