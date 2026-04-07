#!/usr/bin/env python3
"""
12h_donchian_20_1d_volume_v2
Hypothesis: On 12-hour timeframe, use Donchian channel (20-period) breakout for entry, confirmed by daily trend direction and volume spike.
Enter long when price breaks above 20-period Donchian high AND daily trend is up (price > SMA50) AND volume > 2x average.
Enter short when price breaks below 20-period Donchian low AND daily trend is down (price < SMA50) AND volume > 2x average.
Exit when price crosses the midline (average of Donchian bands) or volume drops below average.
Donchian channels capture breakouts with clear levels; daily trend filter ensures alignment with higher timeframe; volume confirms institutional participation.
Target: 20-30 trades/year to minimize fee dust while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1d_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate daily SMA50 for trend filter
    sma50 = pd.Series(d_close).rolling(window=50, min_periods=50).mean().values
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50)
    
    # Volume filter: 12h volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if Donchian or SMA not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(sma50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Daily trend filter
        trend_up = close[i] > sma50_aligned[i]  # Price above daily SMA50
        trend_down = close[i] < sma50_aligned[i]  # Price below daily SMA50
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 2.0
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses below Donchian midline
            if close[i] < donchian_mid[i]:
                exit_long = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses above Donchian midline
            if close[i] > donchian_mid[i]:
                exit_short = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Donchian breakout up AND daily trend up AND volume confirmed
            long_entry = breakout_up and trend_up and vol_confirmed
            
            # Short entry: Donchian breakout down AND daily trend down AND volume confirmed
            short_entry = breakout_down and trend_down and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals