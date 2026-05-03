#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In ranging markets: mean revert from extremes when Williams %R crosses back from overbought/oversold
# In trending markets: only take mean revert trades aligned with 1d EMA50 trend (avoid counter-trend)
# Volume confirmation ensures breakouts have participation, reducing false signals
# 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in both bull/bear: trend filter adapts direction, mean reversion captures swings within trends

name = "6h_WilliamsR_MeanReversion_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Williams %R mean reversion signals with 1d trend filter
        # Long: Williams %R crosses above -80 (exiting oversold) + volume spike + price above 1d EMA50
        # Short: Williams %R crosses below -20 (exiting overbought) + volume spike + price below 1d EMA50
        if position == 0:
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum loss) OR price below 1d EMA50
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum loss) OR price above 1d EMA50
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals