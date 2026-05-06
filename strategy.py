#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume spike and ADX trend filter
# - Uses 1d Donchian channels (20-period) for long-term structure
# - Uses 4h volume spike (2x 10-period average) for entry confirmation
# - Uses 4h ADX > 25 to filter for trending markets only
# - Enters long when price breaks above 1d Donchian upper with volume and trend
# - Enters short when price breaks below 1d Donchian lower with volume and trend
# - Exits when price returns to 1d Donchian middle (median)
# - Position size: 0.25 (25% of capital)
# - Designed to capture major trend moves with institutional level respect
# - Target: 80-180 total trades over 4 years (20-45/year)

name = "4h_1dDonchian_20_Volume_ADX_Trend_v2"
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
    
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2  # Median line for exit
    
    # Align 1d Donchian channels to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_4h = align_htf_to_ltf(prices, df_1d, middle_20)
    
    # Volume filter (4h timeframe) - volume spike > 2x 10-period average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (2.0 * vol_ma_10)
    
    # ADX filter (4h timeframe) - trend strength
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
        
        # Wilder's smoothing
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
    
    adx_values = calculate_adx(high, low, close, 14)
    adx_filter = adx_values > 25  # Strong trend filter
    
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
            # Long: break above 1d Donchian upper with volume and trend
            if close[i] > upper_20_4h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Donchian lower with volume and trend
            elif close[i] < lower_20_4h[i] and volume_spike[i] and adx_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle
            if close[i] < middle_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle
            if close[i] > middle_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals