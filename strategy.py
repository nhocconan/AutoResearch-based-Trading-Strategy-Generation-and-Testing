#!/usr/bin/env python3
name = "1d_WeeklyDonchianBreakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian channels
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    upper_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    lower_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Calculate 50-period daily EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-day average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR > 0.5% of price to avoid low volatility periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.005 * close  # ATR > 0.5% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(upper_20w_aligned[i]) or np.isnan(lower_20w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-day average
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper, above 50-day EMA (uptrend), with volume spike
            buffer = 0.0005 * close[i]  # Small buffer to avoid false breakouts
            if (close[i] > upper_20w_aligned[i] + buffer and 
                close[i] > ema_50_1d_aligned[i] + buffer and   # Daily uptrend
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower, below 50-day EMA (downtrend), with volume spike
            elif (close[i] < lower_20w_aligned[i] - buffer and 
                  close[i] < ema_50_1d_aligned[i] - buffer and   # Daily downtrend
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to weekly Donchian midpoint
            midpoint = (upper_20w_aligned[i] + lower_20w_aligned[i]) / 2
            range_width = upper_20w_aligned[i] - lower_20w_aligned[i]
            # Exit when price reaches midpoint (mean reversion within the weekly range)
            at_midpoint = abs(close[i] - midpoint) < range_width * 0.05  # Within 5% of midpoint
            
            if at_midpoint:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals