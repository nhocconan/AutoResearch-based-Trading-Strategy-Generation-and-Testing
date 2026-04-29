#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 20-period high, 1d EMA34 up-trend, volume > 1.5x average
# Short when price breaks below 20-period low, 1d EMA34 down-trend, volume > 1.5x average
# Exit when price crosses the 20-period midpoint
# Uses discrete position sizing (0.25) and moderate volume filter to target ~15-30 trades/year.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# 12h timeframe reduces trade frequency to minimize fee drag while capturing medium-term trends.

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Get 12h data for Donchian levels (based on previous 20 periods)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h Donchian levels using previous 20 periods (shifted by 1)
    high_20 = df_12h['high'].rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = df_12h['low'].rolling(window=20, min_periods=20).min().shift(1).values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Align 12h indicators to 12h timeframe (no additional delay needed for Donchian)
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_12h, mid_20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 35)  # Volume and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_high20 = high_20_aligned[i]
        curr_low20 = low_20_aligned[i]
        curr_mid20 = mid_20_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 20-period midpoint (mean reversion)
            if curr_close < curr_mid20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 20-period midpoint (mean reversion)
            if curr_close > curr_mid20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average (moderate filter)
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above 20-period high, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_high20 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period low, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_low20 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals