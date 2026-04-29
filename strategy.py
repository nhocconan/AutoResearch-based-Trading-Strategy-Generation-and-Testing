#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above 20-period high AND price > 1d EMA50 AND volume > 2x 20-period average
# Short when price breaks below 20-period low AND price < 1d EMA50 AND volume > 2x 20-period average
# Exit when price returns to 10-period midpoint (mean reversion) or opposite breakout occurs
# Designed for ~15-25 trades/year on 12h timeframe to minimize fee drag
# Uses 1d trend filter to avoid counter-trend trades in strong markets
# Volume spike ensures institutional participation and reduces false breakouts

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 20-period average volume for spike confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        curr_donchian_high = highest_high[i]
        curr_donchian_low = lowest_low[i]
        curr_donchian_mid = donchian_mid[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or opposite breakout
            if curr_close <= curr_donchian_mid or curr_low <= curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or opposite breakout
            if curr_close >= curr_donchian_mid or curr_high >= curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high AND uptrend (price > 1d EMA50)
            if vol_confirm and curr_high > curr_donchian_high and curr_close > curr_ema50_1d:
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian low AND downtrend (price < 1d EMA50)
            elif vol_confirm and curr_low < curr_donchian_low and curr_close < curr_ema50_1d:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals