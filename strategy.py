#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: On 4h timeframe, use 1-day Donchian breakout with trend filter and volume confirmation. 
Enter long when price breaks above 1D Donchian upper (20-period high) with EMA50 > EMA200 and volume > 1.5x average; 
enter short when price breaks below 1D Donchian lower (20-period low) with EMA50 < EMA200 and volume > 1.5x average. 
Exit on opposite Donchian break or trend reversal. Uses daily structure for robustness, works in bull/bear via EMA filter.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA50 and EMA200 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian upper = 20-day high, lower = 20-day low
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (shifted by 1 day for look-ahead prevention)
    donch_high_4h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation (24-period average on 4h = 24*4h = 96h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price breaks below Donchian lower (contrarian exit)
            if close[i] < donch_low_4h[i]:
                exit_long = True
            # Exit if EMA50 crosses below EMA200 (trend reversal)
            elif ema50[i] < ema200[i] and ema50[i-1] >= ema200[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price breaks above Donchian upper (contrarian exit)
            if close[i] > donch_high_4h[i]:
                exit_short = True
            # Exit if EMA50 crosses above EMA200 (trend reversal)
            elif ema50[i] > ema200[i] and ema50[i-1] <= ema200[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with EMA50 > EMA200 and volume confirmation
            long_entry = False
            if (close[i] > donch_high_4h[i] and close[i-1] <= donch_high_4h[i-1] and
                ema50[i] > ema200[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below Donchian lower with EMA50 < EMA200 and volume confirmation
            short_entry = False
            if (close[i] < donch_low_4h[i] and close[i-1] >= donch_low_4h[i-1] and
                ema50[i] < ema200[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals