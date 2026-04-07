#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume and ADX Filter
# Hypothesis: Weekly Donchian breakouts capture institutional moves, while volume confirms participation
# and ADX filters for trending conditions. Works in bull markets (breakouts continue) and bear markets
# (breakdowns continue). The 1d timeframe ensures low trade frequency to minimize drag.
# Target: 20-50 trades per year (80-200 over 4 years).

name = "1d_weekly_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian(20)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate 20-period rolling max/min on weekly data
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    weekly_high_max = weekly_high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_min = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use completed weekly bar only (avoid look-ahead)
    weekly_high_max = np.roll(weekly_high_max, 1)
    weekly_low_min = np.roll(weekly_low_min, 1)
    weekly_high_max[0] = weekly_high_max[1] if len(weekly_high_max) > 1 else 0
    weekly_low_min[0] = weekly_low_min[1] if len(weekly_low_min) > 1 else 0
    
    # Align to daily timeframe
    weekly_high_max_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_max)
    weekly_low_min_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_min)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter: ADX > 25 for trending markets
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[:period])
        minus_dm_sum = np.sum(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx_period = 14
        adx[2*period-1] = np.mean(dx[period-1:2*period-1]) if 2*period-1 < len(dx) else 0
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period if adx[i-1] != 0 else 0
        
        return adx
    
    adx = calculate_adx(high, low, close)
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_high_max_aligned[i]) or np.isnan(weekly_low_min_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to weekly low or ADX weakens
            if (close[i] <= weekly_low_min_aligned[i] or not adx_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to weekly high or ADX weakens
            if (close[i] >= weekly_high_max_aligned[i] or not adx_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above weekly high with volume and trend
            if ((high[i] > weekly_high_max_aligned[i] or close[i] > weekly_high_max_aligned[i]) and 
                vol_filter[i] and adx_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly low with volume and trend
            elif ((low[i] < weekly_low_min_aligned[i] or close[i] < weekly_low_min_aligned[i]) and 
                  vol_filter[i] and adx_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals