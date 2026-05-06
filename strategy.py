#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h ADX for trend filtering with 1d Donchian breakout entries
# - Uses 12h ADX > 25 to filter for strong trends only (avoids whipsaws in ranging markets)
# - Uses 1d Donchian channels (20-period) for entries and exits (captures major trend moves)
# - Uses 4h volume spike (1.5x 20-period average) for entry confirmation
# - Enters long when price breaks above 1d Donchian upper with volume and 12h trend
# - Enters short when price breaks below 1d Donchian lower with volume and 12h trend
# - Exits when price returns to 1d Donchian middle or opposite band
# - Designed to work in both bull and bear markets by only trading strong trends
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_1dDonchian_20_12hADX_Volume"
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2  # Median line for exit
    
    # Align 1d Donchian channels to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_4h = align_htf_to_ltf(prices, df_1d, middle_20)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr = np.zeros_like(high_12h)
    tr[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(high_12h)):
        tr[i] = max(high_12h[i] - low_12h[i], 
                   abs(high_12h[i] - close_12h[i-1]), 
                   abs(low_12h[i] - close_12h[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(high_12h)
    for i in range(1, len(high_12h)):
        plus_dm[i] = max(high_12h[i] - high_12h[i-1], 0)
        minus_dm[i] = max(low_12h[i-1] - low_12h[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
    
    # Wilder's smoothing
    period = 14
    atr = np.zeros_like(high_12h)
    plus_di = np.zeros_like(high_12h)
    minus_di = np.zeros_like(high_12h)
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_sum = np.sum(plus_dm[1:period+1])
    minus_dm_sum = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, len(high_12h)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - plus_dm[i-period] + plus_dm[i]
        minus_dm_sum = minus_dm_sum - minus_dm[i-period] + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
    
    # DX and ADX
    dx = np.zeros_like(high_12h)
    adx_12h = np.zeros_like(high_12h)
    for i in range(2*period, len(high_12h)):
        di_diff = abs(plus_di[i] - minus_di[i])
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
    
    # Smooth DX to get ADX
    adx_12h[2*period] = np.mean(dx[2*period:3*period]) if 3*period <= len(high_12h) else 0
    for i in range(3*period, len(high_12h)):
        adx_12h[i] = (adx_12h[i-1] * (period-1) + dx[i]) / period
    
    # Align 12h ADX to 4h timeframe
    adx_12h_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    adx_filter = adx_12h_4h > 25  # Strong trend filter
    
    # Volume filter (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(middle_20_4h[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 1d Donchian upper with volume and 12h trend
            if close[i] > upper_20_4h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Donchian lower with volume and 12h trend
            elif close[i] < lower_20_4h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle OR breaks below lower band
            if close[i] < middle_20_4h[i] or close[i] < lower_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle OR breaks above upper band
            if close[i] > middle_20_4h[i] or close[i] > upper_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals