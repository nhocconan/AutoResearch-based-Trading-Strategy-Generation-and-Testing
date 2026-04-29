#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume spike
# Long when price breaks above 4h Donchian upper (20), 1d EMA50 up, volume > 1.8x 20-bar avg
# Short when price breaks below 4h Donchian lower (20), 1d EMA50 down, volume > 1.8x 20-bar avg
# Exit on opposite Donchian break (long exits on lower break, short exits on upper break)
# Uses discrete sizing 0.25 and strong volume filter to target 20-50 trades/year.
# Donchian provides structure, EMA50 filters trend direction, volume confirms conviction.
# Works in bull/bear by following higher timeframe trend.

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian upper/lower (20-period) using completed 4h bars only
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_20_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_20_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 4h timeframe (no additional delay needed)
    donchian_20_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_upper)
    donchian_20_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_lower)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_20_upper_aligned[i]) or np.isnan(donchian_20_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_upper = donchian_20_upper_aligned[i]
        curr_donchian_lower = donchian_20_lower_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below 4h Donchian lower (trend reversal)
            if curr_low < curr_donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 4h Donchian upper (trend reversal)
            if curr_high > curr_donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above 4h Donchian upper, 1d EMA50 up-trend, volume confirmed
            if curr_high > curr_donchian_upper and curr_close > curr_ema50_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 4h Donchian lower, 1d EMA50 down-trend, volume confirmed
            elif curr_low < curr_donchian_lower and curr_close < curr_ema50_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals