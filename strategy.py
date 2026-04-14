#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot breakout with 1-day volume confirmation and ADX trend filter
# Long when price breaks above H3 with ADX > 25 and volume > 1.5x 20-period average
# Short when price breaks below L3 with ADX > 25 and volume > 1.5x 20-period average
# Exit when price returns to the Pivot Point level (midpoint)
# Uses Camarilla levels from daily chart for institutional support/resistance
# ADX filter ensures we only trade in trending conditions to avoid whipsaws
# Volume confirmation adds conviction to breakouts
# Target: 50-100 total trades over 4 years (12-25/year) for low frequency and high conviction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.25*(High-Low), etc.
    # L3 = Close - 1.25*(High-Low), L4 = Close - 1.5*(High-Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    h3_1d = close_1d + 1.1 * range_1d * 1.25 / 2  # H3 = C + 1.1*(HL)/2 * 1.25
    l3_1d = close_1d - 1.1 * range_1d * 1.25 / 2  # L3 = C - 1.1*(HL)/2 * 1.25
    pivot_point_1d = pivot_1d  # Pivot point for exit
    
    # Align Camarilla levels to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_point_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_point_1d)
    
    # Calculate ADX on 1d for trend strength
    # ADX requires +DI and -DI calculation
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    # Smooth TR and DM (14-period Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # DI and DX
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    
    # ADX (14-period smoothed DX)
    adx_1d = wilders_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 1d data + ADX smoothing)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(pivot_point_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above H3 + ADX > 25 + volume confirmation
            if (price > h3_1d_aligned[i] and adx_1d_aligned[i] > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below L3 + ADX > 25 + volume confirmation
            elif (price < l3_1d_aligned[i] and adx_1d_aligned[i] > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot point (mean reversion)
            if price <= pivot_point_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot point (mean reversion)
            if price >= pivot_point_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_ADX_Volume"
timeframe = "4h"
leverage = 1.0