#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Williams %R mean reversion + 1d volume spike + ATR filter.
Long when 1w Williams %R < -80 (oversold) and 1d volume > 2.0x 20-period average and price > 12h EMA34 (mild trend filter).
Short when 1w Williams %R > -20 (overbought) and 1d volume > 2.0x 20-period average and price < 12h EMA34.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Williams %R captures extreme reversals; volume confirms participation; EMA34 avoids counter-trend trades.
Designed to work in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Get 12h data for EMA34
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 1w Williams %R (14-period)
    def williams_r(high_vals, low_vals, close_vals, window):
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_vals) / (highest_high - lowest_low)
        # Handle division by zero
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_14_1w = williams_r(high_1w, low_1w, close_1w, 14)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA34
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_34_12h = ema(close_12h, 34)
    
    # Align all to primary timeframe (12h)
    wr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, wr_14_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-day average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + volume + price > EMA34
            if (wr_14_1w_aligned[i] < -80 and 
                volume_confirmed and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + volume + price < EMA34
            elif (wr_14_1w_aligned[i] > -20 and 
                  volume_confirmed and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (recovering from oversold)
            if wr_14_1w_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (declining from overbought)
            if wr_14_1w_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wWilliamsR14_VolumeSpike_EMA34Filter"
timeframe = "12h"
leverage = 1.0