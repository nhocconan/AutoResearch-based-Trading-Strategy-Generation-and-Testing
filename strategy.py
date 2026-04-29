#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h/1d trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) AND 1d EMA50 uptrend AND volume spike
# Short when price breaks below 4h Donchian lower (20) AND 1d EMA50 downtrend AND volume spike
# Exit when price crosses 4h Donchian midpoint OR trend reverses
# Uses 4h for structure/direction, 1h for precise entry timing, volume for momentum validation
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe

name = "1h_Donchian_4h1dTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike confirmation: current volume > 2.0x 24-period average (1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(donchian_mid_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_24[i]
        curr_donch_up = donchian_upper_4h_aligned[i]
        curr_donch_low = donchian_lower_4h_aligned[i]
        curr_donch_mid = donchian_mid_4h_aligned[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 2.0x 24-period average
        vol_spike = curr_vol > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price crosses below 4h Donchian midpoint OR 1d EMA50 turns down
            if curr_close < curr_donch_mid or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h Donchian midpoint OR 1d EMA50 turns up
            if curr_close > curr_donch_mid or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian upper AND 1d EMA50 uptrend AND volume spike
            if (curr_high > curr_donch_up and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian lower AND 1d EMA50 downtrend AND volume spike
            elif (curr_low < curr_donch_low and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals