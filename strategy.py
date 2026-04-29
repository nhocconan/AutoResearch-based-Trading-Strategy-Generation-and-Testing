#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.8x 20-bar average volume
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.8x 20-bar average volume
# Exit when price crosses the 10-bar moving average in the opposite direction
# Uses volume spike and 1d EMA trend filter to reduce false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-period average volume for spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 10, 20)  # 1d EMA34, Donchian(20), EMA10, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ema_10[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_10 = ema_10[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        vol_spike = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price crosses below 10-period EMA
            if curr_close < curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-period EMA
            if curr_close > curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA34 AND volume spike
            if curr_close > curr_donchian_high and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1d EMA34 AND volume spike
            elif curr_close < curr_donchian_low and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals