#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d strategy using 1w Camarilla pivot levels for direction and volume confirmation
    # Works in both bull and bear: Camarilla captures weekly reversals, volume confirms institutional interest
    # Target: 15-25 trades/year to minimize fee drag on 1d timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for primary calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for Camarilla pivot calculation (HTF for direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily OHLC
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get weekly OHLC for Camarilla levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each week (H3 and L3 levels - more conservative than H4/L4)
    camarilla_h3_weekly = weekly_close + (weekly_high - weekly_low) * 1.1 / 4
    camarilla_l3_weekly = weekly_close - (weekly_high - weekly_low) * 1.1 / 4
    
    # Calculate 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 1d primary timeframe
    camarilla_h3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_weekly)
    camarilla_l3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_weekly)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_weekly_aligned[i]) or 
            np.isnan(camarilla_l3_weekly_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-period average
        volume_confirmed = daily_volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Entry conditions: Camarilla level break + volume confirmation
        enter_long = (close[i] > camarilla_h3_weekly_aligned[i]) and volume_confirmed
        enter_short = (close[i] < camarilla_l3_weekly_aligned[i]) and volume_confirmed
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = position == 1 and close[i] < camarilla_l3_weekly_aligned[i]
        exit_short = position == -1 and close[i] > camarilla_h3_weekly_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_pivot_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0