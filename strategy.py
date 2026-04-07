#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 4h timeframe, use Donchian channel (20) breakout with daily EMA trend filter and volume confirmation. 
Enter long when price breaks above 20-period high with daily EMA50 > EMA200 and volume > 1.5x average; 
enter short when price breaks below 20-period low with daily EMA50 < EMA200 and volume > 1.5x average. 
Exit when price crosses the midline (10-period average of high/low) or trend reverses. 
This strategy captures breakouts with trend alignment and volume confirmation, targeting 20-40 trades/year.
Works in bull/bear via daily EMA trend filter. Uses Donchian for clear breakout levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Calculate daily EMA50 and EMA200 for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get daily data for trend filter (using close prices)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 and EMA200 from daily closes
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align daily EMAs to 4h timeframe (shifted by 1 day for look-ahead prevention)
    ema50_4h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    ema200_4h = align_htf_to_ltf(prices, df_1d, daily_ema200)
    
    # Volume confirmation (6-period average on 4h = 1.5 days)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_4h[i]) or np.isnan(ema200_4h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 6-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below midline
            if close[i] < donchian_mid[i]:
                exit_long = True
            # Exit if daily EMA50 crosses below EMA200 (trend reversal)
            elif ema50_4h[i] < ema200_4h[i] and ema50_4h[i-1] >= ema200_4h[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above midline
            if close[i] > donchian_mid[i]:
                exit_short = True
            # Exit if daily EMA50 crosses above EMA200 (trend reversal)
            elif ema50_4h[i] > ema200_4h[i] and ema50_4h[i-1] <= ema200_4h[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with daily EMA50 > EMA200 and volume confirmation
            long_entry = False
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and
                ema50_4h[i] > ema200_4h[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below Donchian low with daily EMA50 < EMA200 and volume confirmation
            short_entry = False
            if (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and
                ema50_4h[i] < ema200_4h[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals