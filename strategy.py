#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot + 1d trend filter + volume confirmation
# Fade at R3/S3 levels in ranging markets (1d ADX < 25), breakout at R4/S4 in trending markets (1d ADX >= 25)
# Uses volume > 1.3x average for confirmation
# Targets 50-150 total trades over 4 years (12-37/year) by using strict pivot levels and regime filter

name = "6h_camarilla_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d ADX for regime detection (trending vs ranging)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d data for Camarilla pivot calculation (use previous day's OHLC)
    # We'll use the previous day's data to avoid look-ahead
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 4
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 4
    camarilla_s4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime: trending (ADX >= 25) or ranging (ADX < 25)
        is_trending = adx_aligned[i] >= 25
        
        if position == 1:  # long position
            # Exit conditions: price reaches S3 (in ranging) or breaks below S4 (in trending)
            if is_trending:
                if close[i] < s4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                if close[i] <= s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions: price reaches R3 (in ranging) or breaks above R4 (in trending)
            if is_trending:
                if close[i] > r4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                if close[i] >= r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Look for entries based on regime
            if is_trending:
                # Trending market: breakout at R4/S4 with volume
                if (close[i] > r4_aligned[i] and volume[i] > volume_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < s4_aligned[i] and volume[i] > volume_threshold[i]):
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging market: fade at R3/S3 with volume
                if (close[i] < r3_aligned[i] and volume[i] > volume_threshold[i]):
                    signals[i] = 0.25  # Long at S3 (fade from R3)
                    position = 1
                elif (close[i] > s3_aligned[i] and volume[i] > volume_threshold[i]):
                    signals[i] = -0.25  # Short at R3 (fade from S3)
                    position = -1
    
    return signals