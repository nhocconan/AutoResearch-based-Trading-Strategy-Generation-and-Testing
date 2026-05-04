#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend + volume spike
# Williams %R(14) identifies overbought/oversold conditions; extreme readings (>80 or <20) with volume spike
# suggest mean reversion entries in the direction of the 1d EMA34 trend (long when price > EMA34 and %R < 20,
# short when price < EMA34 and %R > 80). Uses 1d EMA34 for trend filter to avoid counter-trend trades.
# Volume confirmation (1.5x 20-period EMA) ensures strong participation and reduces false signals.
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in both bull and bear markets by following the higher-timeframe trend and fading extremes only when aligned.

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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # avoid division by zero
        else:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100.0
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams %R mean reversion with 1d EMA34 trend filter
        # Long: %R < 20 (oversold) + volume spike + price above 1d EMA34 (uptrend)
        # Short: %R > 80 (overbought) + volume spike + price below 1d EMA34 (downtrend)
        if position == 0:
            if (williams_r[i] < 20.0 and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (williams_r[i] > 80.0 and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: %R > 50 (exit oversold) OR price below 1d EMA34 (trend change)
            if williams_r[i] > 50.0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R < 50 (exit overbought) OR price above 1d EMA34 (trend change)
            if williams_r[i] < 50.0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals