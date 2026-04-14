#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R for mean reversion and 4h Donchian breakout for entry.
# 12h Williams %R < -80 (oversold) or > -20 (overbought) identifies extremes.
# Entry only when price breaks Donchian(20) channel in the direction of the mean reversion signal.
# Volume confirmation (>1.3x 20-period average) reduces false breakouts.
# ATR-based stop loss manages risk (exit when price moves against position by 2.5x ATR).
# Designed to work in both bull and bear markets by using Williams %R to avoid trend-following in chop.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    willr_period = 14
    highest_high = pd.Series(high_12h).rolling(window=willr_period, min_periods=willr_period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=willr_period, min_periods=willr_period).min().values
    willr = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Load 4h data ONCE for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donch_period = 20
    upper_channel = pd.Series(high_4h).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_channel = pd.Series(low_4h).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Calculate ATR for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to lower timeframe
    willr_aligned = align_htf_to_ltf(prices, df_12h, willr)
    upper_channel_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20, 14)  # Need Donchian, Williams %R, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(willr_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Williams %R signals
        oversold = willr_aligned[i] < -80
        overbought = willr_aligned[i] > -20
        
        if position == 0:
            # Look for Donchian breakouts in direction of Williams %R signal
            # Long: price breaks above upper Donchian AND oversold condition
            if (close[i] > upper_channel_aligned[i] and 
                oversold and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian AND overbought condition
            elif (close[i] < lower_channel_aligned[i] and 
                  overbought and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price moves against position by 2.5x ATR or Donchian breakout in opposite direction
            if (close[i] <= upper_channel_aligned[i] - 2.5 * atr[i] or
                close[i] < lower_channel_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price moves against position by 2.5x ATR or Donchian breakout in opposite direction
            if (close[i] >= lower_channel_aligned[i] + 2.5 * atr[i] or
                close[i] > upper_channel_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hWilliamsR_4hDonchian_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0