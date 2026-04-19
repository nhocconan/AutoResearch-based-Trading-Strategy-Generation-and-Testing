#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In ranging markets, Williams %R extremes indicate mean reversion opportunities
# In trending markets, only take trades in the direction of the weekly trend
# Uses Williams %R(14) for overbought/oversold conditions, weekly EMA(34) for trend,
# and volume confirmation to filter false signals. Designed for low-frequency,
# high-conviction trades on 12h timeframe to minimize fee drag.
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years)

name = "12h_WilliamsR_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    wr = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need Williams %R and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr_val = wr_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R oversold (-80 or below) + above weekly EMA + volume
            if wr_val <= -80 and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (-20 or above) + below weekly EMA + volume
            elif wr_val >= -20 and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or price crosses below weekly EMA
            if wr_val > -50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or price crosses above weekly EMA
            if wr_val < -50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals