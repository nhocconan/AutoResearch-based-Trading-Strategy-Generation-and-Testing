#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for trend direction and Williams %R(14) for oversold/overbought signals
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 12h trend direction and using Williams %R for mean reversion entries
# Prioritizes BTC/ETH performance with SOL as secondary

name = "6h_WilliamsR_MeanReversion_12hEMA50_Trend_Volume"
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
    
    # Get 12h data for Williams %R calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_for_willr = df_12h['close'].values
    
    willr_values = np.full(len(close_12h), np.nan)
    for i in range(13, len(close_12h)):
        highest_high = np.max(high_12h[i-13:i+1])
        lowest_low = np.min(low_12h[i-13:i+1])
        if highest_high != lowest_low:
            willr_values[i] = -100 * (highest_high - close_12h_for_willr[i]) / (highest_high - lowest_low)
        else:
            willr_values[i] = -50  # neutral when range is zero
    
    # Align Williams %R to 6h timeframe (use previous completed 12h bar)
    willr_aligned = align_htf_to_ltf(prices, df_12h, willr_values)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(willr_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams %R mean reversion with 12h trend filter
        # Long: Williams %R < -80 (oversold) + volume spike + price above 12h EMA50 (uptrend)
        # Short: Williams %R > -20 (overbought) + volume spike + price below 12h EMA50 (downtrend)
        if position == 0:
            if (willr_aligned[i] < -80 and volume_spike and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (willr_aligned[i] > -20 and volume_spike and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (exit oversold) OR price below 12h EMA50 (trend change)
            if willr_aligned[i] > -50 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (exit overbought) OR price above 12h EMA50 (trend change)
            if willr_aligned[i] < -50 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals