#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_With_WeeklyTrend_Filter
Hypothesis: Use weekly trend direction to filter Camarilla pivot breakouts on 12h timeframe.
Buy when price breaks above H3 with price above weekly SMA50; short when breaks below L3 with price below weekly SMA50.
Add volume confirmation to ensure institutional participation. Designed for low trade frequency to minimize fee drag
while capturing high-probability breakouts in trending markets and avoiding false signals in ranging conditions.
"""

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
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 12h bar using prior day's data
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    
    for i in range(n):
        # Find prior day's data (same date - 1 day)
        current_date = pd.Timestamp(prices.iloc[i]['open_time']).date()
        prior_date = current_date - pd.Timedelta(days=1)
        
        # Get prior day's OHLC from daily data
        mask = (df_1d['open_time'].dt.date == prior_date)
        if mask.any():
            idx = df_1d.index[mask][0]  # Take first match
            phigh = high_1d[idx]
            plow = low_1d[idx]
            pclose = close_1d[idx]
            
            range_val = phigh - plow
            if range_val > 0:
                camarilla_high[i] = pclose + (range_val * 1.1 / 2)  # H3 level
                camarilla_low[i] = pclose - (range_val * 1.1 / 2)   # L3 level
    
    # Volume confirmation: >1.5x 24-period average (2 days of 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need weekly SMA and prior day data
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_high[i]) or 
            np.isnan(camarilla_low[i]) or
            np.isnan(sma_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_level = camarilla_high[i]
        low_level = camarilla_low[i]
        weekly_trend = sma_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above H3 with weekly uptrend and volume spike
            if price > high_level and price > weekly_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with weekly downtrend and volume spike
            elif price < low_level and price < weekly_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below L3 or weekly trend turns down
            if price < low_level or price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above H3 or weekly trend turns up
            if price > high_level or price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_With_WeeklyTrend_Filter"
timeframe = "12h"
leverage = 1.0