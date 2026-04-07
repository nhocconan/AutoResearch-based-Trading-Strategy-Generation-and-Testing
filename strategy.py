#!/usr/bin/env python3
"""
6h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 6h timeframe, use weekly Donchian breakouts with 1d EMA trend filter and volume confirmation. 
Enter long when price breaks above weekly Donchian high (20-period) with price above 1d EMA and volume > 1.5x average.
Enter short when price breaks below weekly Donchian low with price below 1d EMA and volume confirmation.
Exit when price returns to the midpoint of the Donchian channel or reverses direction.
This strategy captures strong trending moves with institutional participation, reducing false signals.
Works in bull/bear via trend filter and breakout logic, targeting 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1w_trend_volume_v1"
timeframe = "6h"
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
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w data for Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channel (20-period high/low)
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align Donchian levels to 6h timeframe
    donch_high_6h = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_6h = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_6h = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_6h[i]) or np.isnan(donch_high_6h[i]) or 
            np.isnan(donch_low_6h[i]) or np.isnan(donch_mid_6h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from 1d EMA
        uptrend = close[i] > ema_1d_6h[i]
        downtrend = close[i] < ema_1d_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to Donchian midpoint
            if close[i] <= donch_mid_6h[i]:
                exit_long = True
            # Exit if price breaks below Donchian low (reversal)
            elif close[i] < donch_low_6h[i]:
                exit_long = True
            # Exit if trend turns down
            elif downtrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to Donchian midpoint
            if close[i] >= donch_mid_6h[i]:
                exit_short = True
            # Exit if price breaks above Donchian high (reversal)
            elif close[i] > donch_high_6h[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above Donchian high with uptrend and volume confirmation
            if close[i] > donch_high_6h[i] and close[i-1] <= donch_high_6h[i-1]:
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below Donchian low with downtrend and volume confirmation
            if close[i] < donch_low_6h[i] and close[i-1] >= donch_low_6h[i-1]:
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals