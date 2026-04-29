#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian channel breakouts capture momentum in trending markets
# Long when price breaks above 20-period high + price > 1d EMA50 + volume > 2.0x 20-period average
# Short when price breaks below 20-period low + price < 1d EMA50 + volume > 2.0x 20-period average
# Exit when price crosses the midline (10-period average of high/low) or opposite Donchian break
# Works in bull markets via buying breakouts, in bear markets via selling breakdowns
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian20_VolumeConfirmation_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian midline (10-period average of high/low) for exit
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid = (high_10 + low_10) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 20)  # Donchian20, 1d EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_mid = donchian_mid[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline (trend weakening)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline (trend weakening)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above 20-period high + uptrend + volume confirmation
            if (i > start_idx and 
                curr_close > curr_high_20 and  # Breakout above 20-period high
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below 20-period low + downtrend + volume confirmation
            elif (i > start_idx and 
                  curr_close < curr_low_20 and  # Breakdown below 20-period low
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals