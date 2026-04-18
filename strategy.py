#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with daily EMA50 filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets.
# Daily EMA50 ensures trades align with intermediate-term trend to avoid counter-trend trades.
# Volume confirmation filters signals with low participation.
# Designed for low trade frequency (15-35/year) to minimize fee drift in 12h timeframe.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
name = "12h_WilliamsR_DailyEMA50_Volume"
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
    
    # Get daily data for Williams %R and EMA50 (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14-period) on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    # Align Williams %R and EMA50 to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with price above EMA50 and volume
            if vol_confirm and williams_r_aligned[i] < -80 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with price below EMA50 and volume
            elif vol_confirm and williams_r_aligned[i] > -20 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or price crosses below EMA50
            if williams_r_aligned[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or price crosses above EMA50
            if williams_r_aligned[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals