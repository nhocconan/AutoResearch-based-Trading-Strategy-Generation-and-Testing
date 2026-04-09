#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume spike confirmation
# Uses 4h price channel breakouts (Donchian high/low) in direction of 1d ADX > 25
# Volume confirmation requires current volume > 1.8x 30-period average to filter weak breakouts
# Designed for 4h timeframe to target 20-50 trades/year (80-200 over 4 years)
# Works in bull/bear: ADX filter ensures we only trend-follow when trend is strong, avoiding whipsaws in ranging markets
# Exit: price retracement to midpoint of channel OR ADX < 20 (trend weakening)

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(n):
        if i < donchian_period - 1:
            upper_channel[i] = np.nan
            lower_channel[i] = np.nan
        else:
            upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
            lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate midpoint for exit
    midpoint = (upper_channel + lower_channel) / 2.0
    
    # Calculate 30-period average volume for volume spike confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 30:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(adx_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.8x 30-period average
        volume_spike = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below midpoint OR ADX < 20 (trend weakening)
            if close[i] < midpoint[i] or adx_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above midpoint OR ADX < 20 (trend weakening)
            if close[i] > midpoint[i] or adx_4h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume spike and ADX > 25 (strong trend)
            if volume_spike and adx_4h[i] > 25:
                # Long entry: price closes above upper channel (bullish breakout)
                if close[i] > upper_channel[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price closes below lower channel (bearish breakout)
                elif close[i] < lower_channel[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals