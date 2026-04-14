#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND daily ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND daily ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline (10-period average of high/low) in opposite direction
# Donchian channels identify breakouts, ADX confirms trend strength, volume validates momentum
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate ADX on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20 for Donchian/ADX + buffer)
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get ADX values aligned to 4h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
        adx_current = adx_aligned[i]
        
        close_current = close[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        donchian_mid_val = donchian_mid[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above Donchian high + ADX > 25 (trending) + volume confirmation
            if (close_current > donchian_high and 
                adx_current > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low + ADX > 25 (trending) + volume confirmation
            elif (close_current < donchian_low and 
                  adx_current > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if close_current < donchian_mid_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if close_current > donchian_mid_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0