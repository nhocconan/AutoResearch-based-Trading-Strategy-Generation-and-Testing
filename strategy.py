#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX25 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In ranging markets (ADX < 25), fade extremes: short at > -20, long at < -80
# In trending markets (ADX >= 25), breakout continuation: long on break above -20, short on break below -80
# Volume spike (2.0x 24-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via breakout continuation and bear markets via mean reversion in ranges.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX25 for trend regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 25)
    dm_plus_smooth = wilders_smoothing(dm_plus, 25)
    dm_minus_smooth = wilders_smoothing(dm_minus, 25)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0,
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 25)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R (14-period)
    def williams_r(high_arr, low_arr, close_arr, period):
        highest_high = np.full_like(high_arr, np.nan)
        lowest_low = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            highest_high[i] = np.max(high_arr[i-period+1:i+1])
            lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * ((highest_high - close_arr) / (highest_high - lowest_low)), -50)
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*6h = 144h = 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(wr_6h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = wr_6h[i]
        curr_adx = adx_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if curr_volume_spike:
                # Ranging market (ADX < 25): mean reversion at extremes
                if curr_adx < 25:
                    # Long at oversold (< -80)
                    if curr_wr < -80:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                    # Short at overbought (> -20)
                    elif curr_wr > -20:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
                # Trending market (ADX >= 25): breakout continuation
                else:
                    # Long on break above -20 (momentum continuation)
                    if curr_wr > -20:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                    # Short on break below -80 (momentum continuation)
                    elif curr_wr < -80:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions
            if curr_adx < 25:
                # In range: exit when WR returns to -50 (mean reversion complete)
                if curr_wr > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trend: exit when WR crosses below -50 (momentum weakening)
                if curr_wr < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if curr_adx < 25:
                # In range: exit when WR returns to -50 (mean reversion complete)
                if curr_wr < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trend: exit when WR crosses above -50 (momentum weakening)
                if curr_wr > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals