#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d ADX25 Regime Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions (long when %R < -80, short when %R > -20)
# 1d ADX > 25 filters for trending markets (avoid ranging/whipsaw)
# Volume spike (>2.0x 24-bar average) confirms momentum strength
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete position sizing 0.25
# Works in both bull/bear: mean reversion in ranging markets (via %R extremes) + trend filter avoids false signals

name = "6h_WilliamsR_Extreme_1dADX25_Regime_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Williams %R(14) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate ADX(14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # First period DM is 0
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    # Handle division by zero
    di_plus = np.where(atr == 0, 0, di_plus)
    di_minus = np.where(atr == 0, 0, di_minus)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = WilderSmoothing(dx, 14)
    
    # Extract Williams %R signals and ADX regime
    williams_r_signal = williams_r  # Values: 0 to -100
    adx_trending = adx > 25  # ADX > 25 indicates trending market
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_signal)
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    
    # Calculate 6h volume spike: >2.0x 24-bar average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(adx_trending_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions
        williams_r_oversold = williams_r_aligned[i] < -80  # Extremely oversold
        williams_r_overbought = williams_r_aligned[i] > -20  # Extremely overbought
        
        # Trend filter: only trade in trending markets (ADX > 25)
        is_trending = adx_trending_aligned[i] > 0.5  # Boolean array converted to float
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = williams_r_oversold and is_trending and vol_confirm
        short_entry = williams_r_overbought and is_trending and vol_confirm
        
        # Exit conditions: opposite extreme or loss of trend/volume
        long_exit = (williams_r_aligned[i] > -20) or (not is_trending) or (not vol_confirm)
        short_exit = (williams_r_aligned[i] < -80) or (not is_trending) or (not vol_confirm)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals