#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Champaign_Channel_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    Weekly Champaign Channel with 1d EMA trend and volume confirmation.
    - Uses weekly EMA20 for trend direction
    - Champaign Channel (20-period high/low) for breakouts
    - Volume spike filter to avoid false breakouts
    - Target: 10-25 trades/year on 1d timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Champaign Channel and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly Champaign Channel (20-period high/low)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # 20-period rolling high and low
    high_roll = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Champaign Channel levels
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Align Champaign Channel to daily
    upper_channel_daily = align_htf_to_ltf(prices, df_weekly, upper_channel)
    lower_channel_daily = align_htf_to_ltf(prices, df_weekly, lower_channel)
    
    # Weekly EMA20 for trend filter
    ema20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_daily = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Volume spike detection (20-period for daily)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel_daily[i]) or np.isnan(lower_channel_daily[i]) or 
            np.isnan(ema20_daily[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above upper channel with uptrend on weekly, volume spike
            if (close[i] > upper_channel_daily[i] and close[i] > ema20_daily[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower channel with downtrend on weekly, volume spike
            elif (close[i] < lower_channel_daily[i] and close[i] < ema20_daily[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly EMA20
            if close[i] < ema20_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly EMA20
            if close[i] > ema20_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals