#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day ADX trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) AND daily ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when Williams %R crosses below -20 (overbought) AND daily ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when opposite Williams %R threshold is crossed
# Williams %R identifies mean reversion, ADX confirms trend strength, volume validates momentum
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
    
    # Calculate Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    
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
    
    # Start after enough data for calculations (max of 14 for Williams %R/ADX + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(willr[i]) or np.isnan(adx[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get ADX values aligned to 4h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
        adx_current = adx_aligned[i]
        
        willr_current = willr[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Williams %R crosses above -80 (oversold) + ADX > 25 (trending) + volume confirmation
            if (willr_current > -80 and willr[i-1] <= -80 and 
                adx_current > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -20 (overbought) + ADX > 25 (trending) + volume confirmation
            elif (willr_current < -20 and willr[i-1] >= -20 and 
                  adx_current > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -20 (overbought)
            if willr_current < -20 and willr[i-1] >= -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -80 (oversold)
            if willr_current > -80 and willr[i-1] <= -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0