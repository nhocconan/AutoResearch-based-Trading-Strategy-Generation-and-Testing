#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; mean reversion in 6h timeframe with 1d trend filter
# Volume spike (2.0x 20 EMA) confirms strong participation to reduce false signals
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe

name = "6h_WilliamsR_MeanReversion_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        highest_high = np.max(high_1d[i-14:i+1])
        lowest_low = np.min(low_1d[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams %R mean reversion with 1d trend filter
        # Long: Williams %R < -80 (oversold) + volume spike + price above 1d EMA34 (uptrend)
        # Short: Williams %R > -20 (overbought) + volume spike + price below 1d EMA34 (downtrend)
        if position == 0:
            if (williams_r_aligned[i] < -80 and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (williams_r_aligned[i] > -20 and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return to neutral) OR price below 1d EMA34 (trend change)
            if williams_r_aligned[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return to neutral) OR price above 1d EMA34 (trend change)
            if williams_r_aligned[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals