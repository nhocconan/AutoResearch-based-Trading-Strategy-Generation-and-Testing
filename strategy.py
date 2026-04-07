#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use 1-day Donchian breakout for trend direction with volume confirmation and 1-week trend filter. Enter long when price breaks above 1-day Donchian upper channel with 1-week uptrend and volume confirmation; enter short when price breaks below 1-day Donchian lower channel with 1-week downtrend and volume confirmation. Exit when price returns to the Donchian middle. This strategy targets strong trending moves with volume confirmation, reducing false signals and trade frequency. Works in bull/bear via weekly trend filter and breakout logic. Adjusted to reduce trade frequency and improve robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1w EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_12h = align_htf_to_ltf(prices, df_1d, donch_mid)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or
            np.isnan(donch_mid_12h[i]) or np.isnan(ema_1w_12h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from 1w EMA
        uptrend = close[i] > ema_1w_12h[i]
        downtrend = close[i] < ema_1w_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to Donchian middle
            if close[i] <= donch_mid_12h[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend and close[i] < donch_high_12h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to Donchian middle
            if close[i] >= donch_mid_12h[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend and close[i] > donch_low_12h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above Donchian upper channel with uptrend and volume confirmation
            if close[i] > donch_high_12h[i] and close[i-1] <= donch_high_12h[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below Donchian lower channel with downtrend and volume confirmation
            if close[i] < donch_low_12h[i] and close[i-1] >= donch_low_12h[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals