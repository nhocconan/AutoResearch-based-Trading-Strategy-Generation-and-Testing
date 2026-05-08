#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w Trend Filter and Volume Confirmation
# Williams %R identifies overbought/oversold conditions. Readings below -80 indicate oversold,
# above -20 indicate overbought. We buy when %R crosses above -80 from below in an uptrend,
# and sell when %R crosses below -20 from above in a downtrend. 1w ADX > 25 ensures we only
# trade in strong trends, avoiding whipsaws in ranges. Volume spike confirms conviction.
# This oscillator-based approach works in both bull and bear markets by capturing
# momentum extremes within strong trends. Targets 15-25 trades per year (~60-100 total).

name = "12h_WilliamsR_1wTrend_Volume"
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
    
    # Get 1w data for Williams %R, ADX, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams %R (14-period) on weekly
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-(period-1):i+1])
            lowest_low[i] = np.min(low[i-(period-1):i+1])
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high_1w, low_1w, close_1w, 14)
    
    # Calculate ADX (14-period) on weekly
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    # Volume spike detection on weekly (24-period MA for ~12 days)
    vol_ma = pd.Series(volume_1w).rolling(window=24, min_periods=24).mean()
    vol_spike_1w = volume_1w > (vol_ma.values * 2.0)
    
    # Align indicators to 12h timeframe
    wr_12h = align_htf_to_ltf(prices, df_1w, wr)
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    vol_spike_12h = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Define overbought/oversold levels
    oversold = -80
    overbought = -20
    strong_trend = adx_12h > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient data for weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_12h[i]) or np.isnan(adx_12h[i]) or np.isnan(vol_spike_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_prev = wr_12h[i-1]
        wr_curr = wr_12h[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below, with volume spike and strong trend
            if wr_prev <= oversold and wr_curr > oversold and vol_spike_12h[i] and strong_trend[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above, with volume spike and strong trend
            elif wr_prev >= overbought and wr_curr < overbought and vol_spike_12h[i] and strong_trend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -20 (overbought) or trend weakens
            if wr_curr >= overbought or not strong_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -80 (oversold) or trend weakens
            if wr_curr <= oversold or not strong_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals