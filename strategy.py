#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Pivot Breakout with Volume and ADX Filter
# Hypothesis: Daily pivot levels act as strong support/resistance. Breaking above R1 or below S1
# with volume confirmation and trending market (ADX > 20) indicates momentum continuation.
# Works in both bull/bear markets: breaks above R1 in bull, breaks below S1 in bear.
# Target: 20-50 trades/year (80-200 over 4 years) to avoid overtrading.

name = "4h_daily_pivot_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    
    # Align daily pivots to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ADX filter: ADX > 20 for trending market
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    high_diff[high_diff < 0] = 0
    low_diff[low_diff > 0] = 0
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((-low_diff > high_diff) & (low_diff < 0), -low_diff, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0]
    tr3[0] = low[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    dx = np.zeros_like(tr)
    adx = np.zeros_like(tr)
    
    period = 14
    alpha = 1.0 / period
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
        plus_dm_sum = (1 - alpha) * plus_dm_sum + alpha * plus_dm[i]
        minus_dm_sum = (1 - alpha) * minus_dm_sum + alpha * minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] > 0 else 0
        dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) > 0 else 0
    
    # Smooth DX to get ADX
    adx[period-1] = np.mean(dx[period-1:2*period-1]) if 2*period-1 < n else 0
    for i in range(2*period-1, n):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    adx_filter = adx > 20
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily pivot or ADX weakens
            if close[i] < pivot_aligned[i] or not adx_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above daily pivot or ADX weakens
            if close[i] > pivot_aligned[i] or not adx_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume and ADX confirmation
            if close[i] > r1_aligned[i] and vol_filter[i] and adx_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume and ADX confirmation
            elif close[i] < s1_aligned[i] and vol_filter[i] and adx_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals