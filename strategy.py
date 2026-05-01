#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA50 for major trend direction to capture multi-week regimes.
# Long when: price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-period average volume.
# Short when: price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-period average volume.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 15-30 trades/year.
# Donchian channels provide clear structural breakouts; 1w EMA50 filters counter-trend noise;
# Volume confirmation ensures breakouts have conviction. Works in bull (trend continuation) and bear (trend acceleration).

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    def rolling_max(arr, window):
        """Calculate rolling maximum"""
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        """Calculate rolling minimum"""
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 20-period average volume for confirmation
    def rolling_mean(arr, window):
        """Calculate rolling mean"""
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmean(arr[i-window+1:i+1])
        return result
    
    avg_volume_20 = rolling_mean(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_avg_volume = avg_volume_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = curr_volume > (1.5 * curr_avg_volume)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND above 1w EMA50 AND volume confirmed
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_50_1w and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA50 AND volume confirmed
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR below 1w EMA50
            if (curr_close < curr_donchian_low or 
                curr_close < curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR above 1w EMA50
            if (curr_close > curr_donchian_high or 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals