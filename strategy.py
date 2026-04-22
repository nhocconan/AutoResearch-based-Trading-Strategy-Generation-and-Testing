#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (below -80 for long, above -20 for short)
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges
# Volume spike confirms momentum behind the move
# Target: 20-30 trades/year per symbol (80-120 total) to avoid fee drag
# Works in both bull and bear markets by only trading with the daily trend

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Williams %R calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # ADX calculation (14-period) for trend strength
    # +DM = max(0, High_t - High_{t-1})
    # -DM = max(0, Low_{t-1} - Low_t)
    # TR = max(High-Low, |High-Prev Close|, |Low-Prev Close|)
    # +DM14 = smoothed +DM, -DM14 = smoothed -DM, TR14 = smoothed TR
    # +DI14 = 100 * +DM14 / TR14, -DI14 = 100 * -DM14 / TR14
    # DX = 100 * |+DI14 - -DI14| / (+DI14 + -DI14)
    # ADX = smoothed DX
    
    # Calculate +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])  # negative of diff so low_{t-1} - low_t
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]  # avoid NaN on first element
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close)
    tr3 = np.abs(low_1d - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            # Wilder's smoothing: today = alpha * current + (1-alpha) * yesterday
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    tr14 = wilders_smoothing(tr, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R below -80 (oversold) + ADX > 25 (trending) + volume spike
            if (williams_r_aligned[i] < -80 and adx_aligned[i] > 25 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R above -20 (overbought) + ADX > 25 (trending) + volume spike
            elif (williams_r_aligned[i] > -20 and adx_aligned[i] > 25 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral territory (-50) or ADX weakens
            if position == 1:
                if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_ADX25_Volume_Spike_Session"
timeframe = "4h"
leverage = 1.0