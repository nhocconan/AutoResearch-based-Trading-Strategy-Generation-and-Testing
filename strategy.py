#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation.
# Long when price breaks above R3 AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
# Short when price breaks below S3 AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
# Exit when price returns to Camarilla H4/L4 levels OR ADX < 20 (ranging).
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide intraday support/resistance, ADX filters for trending markets only, volume confirms participation.
# Designed to work in both bull (breakouts with volume) and bear (breakdowns with volume) markets.

name = "12h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # But we use R3 and S3 for breakouts, H4 and L4 for exits
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.125 * range_1d
    s3_1d = close_1d - 1.125 * range_1d
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate 1d ADX (trend strength filter)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 1d ADX conditions
        adx_trending = adx_1d_aligned[i] > 25
        adx_ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: price breaks above R3 AND trending AND volume spike AND session
            if close[i] > r3_1d_aligned[i] and adx_trending and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND trending AND volume spike AND session
            elif close[i] < s3_1d_aligned[i] and adx_trending and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below H4 OR ADX becomes ranging
            if close[i] < h4_1d_aligned[i] or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above L4 OR ADX becomes ranging
            if close[i] > l4_1d_aligned[i] or adx_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals