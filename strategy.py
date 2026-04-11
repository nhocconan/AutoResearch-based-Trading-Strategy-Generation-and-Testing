#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume spike and 1d ADX trend filter
# Long when price crosses above Camarilla L3 (support) with volume > 2x average and ADX < 20 (range)
# Short when price crosses below Camarilla H3 (resistance) with volume > 2x average and ADX < 20 (range)
# Exit when price reaches Camarilla H3 (for longs) or L3 (for shorts) or ADX > 25 (trend onset)
# Designed for 15-35 trades/year on 12h timeframe with mean reversion in ranging markets

name = "12h_1d_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.zeros_like(dx)
    adx[13] = np.mean(dx[:14])  # First ADX at index 13 (14-period)
    for i in range(14, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 12h timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d average volume for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla levels use previous day's close, high, low
    close_1d_prev = np.concatenate([[close_1d[0]], close_1d[:-1]])  # Previous day close
    high_1d_prev = np.concatenate([[high_1d[0]], high_1d[:-1]])    # Previous day high
    low_1d_prev = np.concatenate([[low_1d[0]], low_1d[:-1]])      # Previous day low
    
    # Typical price for pivot calculation
    typical_price = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    range_1d = high_1d_prev - low_1d_prev
    
    # Camarilla levels
    camarilla_h3 = typical_price + range_1d * 1.1 / 4
    camarilla_l3 = typical_price - range_1d * 1.1 / 4
    camarilla_h4 = typical_price + range_1d * 1.1 / 2
    camarilla_l4 = typical_price - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 2x 20-period average
        volume_spike = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # Range condition: ADX < 20 (no strong trend)
        is_range = adx_14_1d_aligned[i] < 20
        
        # Trend onset: ADX > 25 (trend starting)
        is_trend = adx_14_1d_aligned[i] > 25
        
        # Entry conditions: price crosses Camarilla L3/H3 with volume spike in ranging market
        long_entry = (close[i-1] <= camarilla_l3_aligned[i-1] and close[i] > camarilla_l3_aligned[i-1]) and volume_spike and is_range
        short_entry = (close[i-1] >= camarilla_h3_aligned[i-1] and close[i] < camarilla_h3_aligned[i-1]) and volume_spike and is_range
        
        # Exit conditions: price reaches opposite Camarilla level or trend starts
        long_exit = (close[i] >= camarilla_h3_aligned[i]) or is_trend
        short_exit = (close[i] <= camarilla_l3_aligned[i]) or is_trend
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals