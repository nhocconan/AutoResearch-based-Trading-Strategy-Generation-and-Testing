#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 1d Camarilla pivot breakout + volume confirmation
    # Long: Williams %R < -80 (oversold) AND price breaks above Camarilla R3 (1d) AND volume > 1.5x avg
    # Short: Williams %R > -20 (overbought) AND price breaks below Camarilla S3 (1d) AND volume > 1.5x avg
    # Exit: Williams %R returns to neutral range (-50 to -30 for longs, -70 to -50 for shorts) OR volume dry-up
    # Using 6h primary timeframe for lower trade frequency, Williams %R for momentum extremes,
    # 1d Camarilla pivot for institutional levels, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align daily Camarilla levels to 6h
    h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get weekly data for Williams %R regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Williams %R(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, (highest_high - close_1w) / hh_ll * -100, -50)
    
    # Align weekly Williams %R to 6h
    williams_r_6h = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Get 6h volume for confirmation (>1.5x 24-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_6h[i]) or np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extremes
        williams_oversold = williams_r_6h[i] < -80
        williams_overbought = williams_r_6h[i] > -20
        williams_neutral_long = williams_r_6h[i] > -50  # Exit long when above -50
        williams_neutral_short = williams_r_6h[i] < -70  # Exit short when below -70
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Williams extreme + Camarilla breakout + volume confirmation
        long_entry = williams_oversold and (close[i] > h3_6h[i]) and vol_confirm
        short_entry = williams_overbought and (close[i] < l3_6h[i]) and vol_confirm
        
        # Exit logic: Williams returns to neutral OR volume dry-up
        long_exit = williams_neutral_long or not vol_confirm
        short_exit = williams_neutral_short or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_williamsr_extreme_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0