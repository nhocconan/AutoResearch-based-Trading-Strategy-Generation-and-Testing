#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and ADX trend filter.
# Uses 12h price relative to Alligator jaws/teeth/lips for trend direction.
# Confirmed by 1d volume > 1.5x 20-period average and 1d ADX > 25 for trend strength.
# In trending markets, follows Alligator alignment (long when green, short when red).
# In ranging markets (ADX < 20), uses mean reversion at Bollinger Bands (20,2).
# Designed to work in both bull and bear markets by adapting to trend strength.
# Target: 15-35 trades/year (60-140 total over 4 years).

name = "12h_WilliamsAlligator_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr[i] = high_1d[i] - low_1d[i]
        else:
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # Directional Movement
    dm_plus = np.zeros(len(df_1d))
    dm_minus = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smoothed values
    atr = np.zeros(len(df_1d))
    dm_plus_smooth = np.zeros(len(df_1d))
    dm_minus_smooth = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 14:
            if i == 0:
                atr[i] = tr[i]
                dm_plus_smooth[i] = dm_plus[i]
                dm_minus_smooth[i] = dm_minus[i]
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.zeros(len(df_1d))
    di_minus = np.zeros(len(df_1d))
    dx = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (smoothed DX)
    adx = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 27:  # need 14+14 periods for smoothing
            adx[i] = np.nan
        elif i == 27:
            adx[i] = np.mean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d data to 12h timeframe
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Williams Alligator (13,8,5)
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw = np.full(n, np.nan)
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth = np.full(n, np.nan)
    # Lips (5-period SMMA, shifted 3 bars)
    lips = np.full(n, np.nan)
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts
    for i in range(8, n):
        jaw[i] = jaw_raw[i-8]
    for i in range(5, n):
        teeth[i] = teeth_raw[i-5]
    for i in range(3, n):
        lips[i] = lips_raw[i-3]
    
    # Calculate Bollinger Bands (20,2) for ranging market
    close_pd = pd.Series(close)
    bb_middle = close_pd.rolling(window=20, min_periods=20).mean().values
    bb_std = close_pd.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 1d bar's data (last completed 1d bar)
        idx_1d = 0
        while idx_1d < len(df_1d) and df_1d.iloc[idx_1d]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1d += 1
        idx_1d -= 1  # last completed 1d bar
        
        if idx_1d < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_avg_20_current = vol_avg_20[idx_1d]
        adx_current = adx[idx_1d]
        
        if np.isnan(vol_avg_20_current) or np.isnan(adx_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_current = df_1d['volume'].iloc[idx_1d]
        vol_confirmed = vol_current > 1.5 * vol_avg_20_current
        
        # Trend detection
        is_trending = adx_current > 25
        is_ranging = adx_current < 20
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                if is_trending:
                    # In trending market: Alligator alignment
                    # Green: lips > teeth > jaw (bullish alignment)
                    # Red: lips < teeth < jaw (bearish alignment)
                    if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                        signals[i] = 0.25
                        position = 1
                    elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                        signals[i] = -0.25
                        position = -1
                elif is_ranging:
                    # In ranging market: mean reversion at Bollinger Bands
                    if close[i] <= bb_lower[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] >= bb_upper[i]:
                        signals[i] = -0.25
                        position = -1
                else:
                    # Transition zone: wait for clearer signal
                    pass
        elif position == 1:
            # Manage long position
            exit_signal = False
            if is_trending:
                # Exit when Alligator alignment breaks (red formation)
                if lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    exit_signal = True
            elif is_ranging:
                # Exit when price reaches middle Bollinger Band
                if close[i] >= bb_middle[i]:
                    exit_signal = True
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            if is_trending:
                # Exit when Alligator alignment breaks (green formation)
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    exit_signal = True
            elif is_ranging:
                # Exit when price reaches middle Bollinger Band
                if close[i] <= bb_middle[i]:
                    exit_signal = True
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals