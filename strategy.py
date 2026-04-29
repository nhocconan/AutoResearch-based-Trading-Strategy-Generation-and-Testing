#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period high, 1w EMA50 up-trend, volume > 2.0x average
# Short when price breaks below 20-period low, 1w EMA50 down-trend, volume > 2.0x average
# Exit when price reverts to 20-period midpoint (mean reversion)
# Uses discrete position sizing (0.25) and tight volume filter to limit trades to 7-25/year.
# Uses 1w for signal direction/trend, 1d only for entry timing and breakout levels.
# Targets 30-100 total trades over 4 years to avoid fee drag while capturing strong breakouts.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Get 1d data for Donchian levels (based on previous 20 periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d Donchian levels using previous 20 periods (shifted by 1)
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1).values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Align 1d indicators to 1d timeframe (no additional delay needed for Donchian)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
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
        curr_ema50_1w = ema_50_1w_aligned[i]
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
            # Volume confirmation: current volume > 2.0x 20-period average (tight filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above 20-period high, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_high20 and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period low, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_low20 and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals