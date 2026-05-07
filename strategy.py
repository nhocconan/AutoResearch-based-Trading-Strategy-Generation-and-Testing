#!/usr/bin/env python3
name = "1d_WeeklyDonchianBreakout_TrendFilter_v3"
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
    
    # Get weekly data for trend filter and Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over past 20 weekly periods
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 weekly periods
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Volume filter: 20-day average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility (ATR > 0.5% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.005 * close  # ATR > 0.5% of price
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure weekly Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.2 x 20-day average
        volume_spike = volume[i] > 2.2 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above weekly upper Donchian, above weekly EMA20 (uptrend), with volume spike
            buffer = 0.001 * close[i]  # 0.1% buffer to avoid whipsaws
            if (close[i] > upper_20_aligned[i] + buffer and 
                close[i] > ema_20_1w_aligned[i] + buffer and   # weekly uptrend
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly lower Donchian, below weekly EMA20 (downtrend), with volume spike
            elif (close[i] < lower_20_aligned[i] - buffer and 
                  close[i] < ema_20_1w_aligned[i] - buffer and   # weekly downtrend
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to weekly Donchian midpoint
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            range_width = upper_20_aligned[i] - lower_20_aligned[i]
            # Exit when within 25% of midpoint (tight exit to manage risk)
            at_mid = abs(close[i] - midpoint) < range_width * 0.25
            
            if at_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals