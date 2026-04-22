#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
# Donchian(20) breakouts capture momentum in both bull and bear markets.
# 1d ADX > 25 ensures we only trade in trending markets, reducing whipsaws.
# Volume confirmation requires current volume > 1.3x 20-period average to validate breakouts.
# Designed for 12h timeframe to target 12-37 trades/year with strict entry conditions.
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
    else:
        adx = np.full_like(high_1d, np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 12h data for Donchian channel (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h data
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    highest_high_aligned = align_htf_to_ltf(prices, df_12h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_12h, lowest_low)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        upper_channel = highest_high_aligned[i]
        lower_channel = lowest_low_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        # ADX filter: trending market (ADX > 25)
        trending = adx_val > 25
        
        if position == 0:
            # Long conditions: breakout above upper channel + trending + volume spike
            if price > upper_channel and trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below lower channel + trending + volume spike
            elif price < lower_channel and trending and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle of channel or trend weakens
                mid_channel = (upper_channel + lower_channel) / 2
                if price < mid_channel or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle of channel or trend weakens
                mid_channel = (upper_channel + lower_channel) / 2
                if price > mid_channel or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0