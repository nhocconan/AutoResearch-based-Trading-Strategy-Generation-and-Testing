#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Donchian channels from 12h price action provide structural breakouts, 1w EMA50 ensures alignment with weekly trend, volume spike confirms breakout validity
# Works in bull markets via upside breaks at upper channel, in bear markets via downside breaks at lower channel
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_Donchian20_VolumeConfirmation_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 1w EMA50, Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average (balanced threshold)
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + uptrend + volume confirmation
            if (curr_close > curr_upper and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + downtrend + volume confirmation
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals