#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use 1w Donchian breakout for trend following with volume confirmation.
Enter long when price breaks above 20-day high with 1w uptrend and volume above average;
enter short when price breaks below 20-day low with 1w downtrend and volume above average.
Exit when price returns to 10-day midpoint or opposite breakout occurs.
This strategy targets major trend moves with volume confirmation, reducing false signals.
Works in bull/bear via 1w trend filter and breakout logic. Designed for low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
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
    
    # 1w data for Donchian and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1w data (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 20-period high and low for Donchian
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # 10-period midpoint for exit
    high_10 = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    midpoint_10 = (high_10 + low_10) / 2
    
    # 1w EMA for trend filter (50-period)
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 1d timeframe
    high_20_1d = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_1d = align_htf_to_ltf(prices, df_1w, low_20)
    midpoint_10_1d = align_htf_to_ltf(prices, df_1w, midpoint_10)
    ema_1w_1d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20_1d[i]) or np.isnan(low_20_1d[i]) or
            np.isnan(midpoint_10_1d[i]) or np.isnan(ema_1w_1d[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from 1w EMA
        uptrend = close[i] > ema_1w_1d[i]
        downtrend = close[i] < ema_1w_1d[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to 10-day midpoint
            if close[i] <= midpoint_10_1d[i]:
                exit_long = True
            # Exit if price breaks below 20-day low (strong reversal)
            elif close[i] < low_20_1d[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to 10-day midpoint
            if close[i] >= midpoint_10_1d[i]:
                exit_short = True
            # Exit if price breaks above 20-day high (strong reversal)
            elif close[i] > high_20_1d[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above 20-day high with uptrend and volume confirmation
            if close[i] > high_20_1d[i] and close[i-1] <= high_20_1d[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below 20-day low with downtrend and volume confirmation
            if close[i] < low_20_1d[i] and close[i-1] >= low_20_1d[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals