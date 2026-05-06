#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R3/S3) with 1d volume spike and 1d ADX trend filter
# - Uses 1d Camarilla R3/S3 levels for entry signals (institutional pivot levels)
# - Uses 1d volume spike (>2.0x 10-period MA) for confirmation
# - Uses 1d ADX > 25 to filter for trending markets only
# - Enters long when price closes above 1d R3 with volume and trend
# - Enters short when price closes below 1d S3 with volume and trend
# - Exits when price returns to 1d Pivot point (mean reversion to mean)
# - Designed to capture institutional level reactions with trend confirmation
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing
# - Works in bull (breakouts hold) and bear (rejections at S3) due to trend filter

name = "12h_1dCamarilla_R3S3_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for 20-period calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels for current day based on previous day's data
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    # Pivot = (H + L + C) / 3
    r3 = close_1d[:-1] + range_1d[:-1] * 1.1 / 2  # Previous day's R3
    s3 = close_1d[:-1] - range_1d[:-1] * 1.1 / 2  # Previous day's S3
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3  # Previous day's pivot
    
    # Shift to align with current day (today's levels based on yesterday's data)
    r3 = np.concatenate([np.array([np.nan]), r3])  # First day has no previous day
    s3 = np.concatenate([np.array([np.nan]), s3])
    pivot = np.concatenate([np.array([np.nan]), pivot])
    
    # Volume spike filter (1d timeframe)
    vol_ma_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_10)  # Strong volume confirmation
    
    # ADX filter (1d timeframe) - trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm[i-period+1] if i-period+1 >= 0 else 0) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm[i-period+1] if i-period+1 >= 0 else 0) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        # Smooth DX to get ADX
        adx[2*period-1] = np.mean(dx[2*period-1:3*period]) if 3*period <= len(high) else 0
        for i in range(3*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_values = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_filter = adx_values > 25  # Strong trend filter
    
    # Align all 1d indicators to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    volume_spike_12h = align_htf_to_ltf(prices, df_1d, volume_spike)
    adx_filter_12h = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(pivot_12h[i]) or 
            np.isnan(volume_spike_12h[i]) or np.isnan(adx_filter_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above 1d R3 with volume and trend
            if close[i] > r3_12h[i] and volume_spike_12h[i] and adx_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below 1d S3 with volume and trend
            elif close[i] < s3_12h[i] and volume_spike_12h[i] and adx_filter_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot (mean reversion)
            if close[i] < pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot (mean reversion)
            if close[i] > pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals