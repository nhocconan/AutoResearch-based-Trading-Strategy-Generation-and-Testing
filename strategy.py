#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R with daily volume confirmation and weekly ADX trend filter
# Long when Williams %R crosses above -20 from below + daily volume > 1.5x 20-period SMA + weekly ADX > 25
# Short when Williams %R crosses below -80 from above + daily volume > 1.5x 20-period SMA + weekly ADX > 25
# Exit when Williams %R crosses opposite threshold (-80 for long, -20 for short) or weekly ADX falls below 20
# Williams %R identifies overbought/oversold conditions, volume confirms momentum, ADX filters for trending markets
# Targets 15-25 trades/year to minimize fee decay while capturing mean reversion in ranging markets and momentum in trending markets

name = "12h_WilliamsR_DailyVolume_WeeklyADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate 12-period Williams %R
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Neutral when no range
    
    # Calculate weekly ADX(14) for trend strength
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((weekly_high - np.roll(weekly_high, 1)) > (np.roll(weekly_low, 1) - weekly_low), 
                       np.maximum(weekly_high - np.roll(weekly_high, 1), 0), 0)
    dm_minus = np.where((np.roll(weekly_low, 1) - weekly_low) > (weekly_high - np.roll(weekly_high, 1)), 
                        np.maximum(np.roll(weekly_low, 1) - weekly_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder smoothing)
    def smooth_series(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI values
    di_plus = np.where(atr > 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr > 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = smooth_series(dx, 14)
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = smooth_series(daily_volume, 20)
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Align daily volume MA to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day SMA
        # Find the most recent completed daily bar
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        vol_filter = False
        if idx_daily >= 0:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for Williams %R signals with volume confirmation and strong trend
            # Long: Williams %R crosses above -20 from below (bullish momentum)
            if i > 0 and williams_r[i-1] <= -20 and williams_r[i] > -20 and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R crosses below -80 from above (bearish momentum)
            elif i > 0 and williams_r[i-1] >= -80 and williams_r[i] < -80 and adx_aligned[i] > 25:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -80 or ADX falls below 20
            if williams_r[i] < -80 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -20 or ADX falls below 20
            if williams_r[i] > -20 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals