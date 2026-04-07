#!/usr/bin/env python3
"""
12h_donchian_20_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use Donchian channel breakout (20) with 1-week trend filter (EMA25 > EMA100) and volume confirmation (>1.5x 20-bar average). 
Enter long when price breaks above Donchian upper band and weekly trend is up with volume confirmation.
Enter short when price breaks below Donchian lower band and weekly trend is down with volume confirmation.
Exit when price crosses the Donchian middle (20-bar average) or volume drops below average.
Targets 12-37 trades/year to minimize fee drag and works in both bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA25 and EMA100 on weekly close
    weekly_close = df_1w['close'].values
    weekly_close_s = pd.Series(weekly_close)
    ema25_1w = weekly_close_s.ewm(span=25, min_periods=25, adjust=False).mean().values
    ema100_1w = weekly_close_s.ewm(span=100, min_periods=100, adjust=False).mean().values
    
    # Align to 12h timeframe (shifted by 1 week to avoid look-ahead)
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after Donchian20 warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema25_1w_aligned[i]) or np.isnan(ema100_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from 1w: up if EMA25 > EMA100, down if EMA25 < EMA100
        trend_up = ema25_1w_aligned[i] > ema100_1w_aligned[i]
        trend_down = ema25_1w_aligned[i] < ema100_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses below Donchian middle
            if close[i] < donchian_mid[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses above Donchian middle
            if close[i] > donchian_mid[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, weekly trend up, volume confirmation
            long_entry = (close[i] > donchian_high[i]) and trend_up and vol_confirm
            
            # Short entry: price breaks below Donchian low, weekly trend down, volume confirmation
            short_entry = (close[i] < donchian_low[i]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals