#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot Range Breakout with Volume Confirmation
# Hypothesis: Weekly pivot levels act as strong support/resistance. Breaking above/below
# the weekly pivot range with volume confirmation captures institutional flow.
# Uses 1d timeframe with 1h pivot calculation for precision. Designed for low turnover
# (target: 15-25 trades/year) to minimize fee drag in ranging/bear markets.

name = "1d_weekly_pivot_range_breakout_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for pivot calculation (more precise than daily)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    # Resample 1h to weekly using actual weekly boundaries
    df_1h_copy = df_1h.copy()
    df_1h_copy['date'] = pd.to_datetime(df_1h_copy['open_time'])
    df_1h_copy.set_index('date', inplace=True)
    
    # Get weekly OHLC
    weekly = df_1h_copy.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate pivot points for each week (using prior week's data)
    weekly['pivot'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
    weekly['support1'] = 2 * weekly['pivot'] - weekly['high'].shift(1)
    weekly['resistance1'] = 2 * weekly['pivot'] - weekly['low'].shift(1)
    weekly['support2'] = weekly['pivot'] - (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['resistance2'] = weekly['pivot'] + (weekly['high'].shift(1) - weekly['low'].shift(1))
    
    # Forward fill weekly levels to 1h
    weekly_cols = ['pivot', 'support1', 'resistance1', 'support2', 'resistance2']
    for col in weekly_cols:
        df_1h_copy[col] = weekly[col].reindex(df_1h_copy.index, method='ffill')
    
    # Align to 1d timeframe
    pivot_1d = align_htf_to_ltf(prices, df_1h_copy, df_1h_copy['pivot'].values)
    support1_1d = align_htf_to_ltf(prices, df_1h_copy, df_1h_copy['support1'].values)
    resistance1_1d = align_htf_to_ltf(prices, df_1h_copy, df_1h_copy['resistance1'].values)
    support2_1d = align_htf_to_ltf(prices, df_1h_copy, df_1h_copy['support2'].values)
    resistance2_1d = align_htf_to_ltf(prices, df_1h_copy, df_1h_copy['resistance2'].values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(pivot_1d[i]) or np.isnan(support1_1d[i]) or 
            np.isnan(resistance1_1d[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price drops below pivot or volume dries up
            if close[i] < pivot_1d[i] or vol_ratio[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above pivot or volume dries up
            if close[i] > pivot_1d[i] or vol_ratio[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: break above resistance1 with volume confirmation
            if (close[i] > resistance1_1d[i] and 
                vol_ratio[i] > 1.5):
                position = 1
                signals[i] = 0.25
            # Short: break below support1 with volume confirmation
            elif (close[i] < support1_1d[i] and 
                  vol_ratio[i] > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals