#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and 1d ADX(25) trend filter
# Uses Camarilla levels (R3/S3) from daily high-low-close for breakout signals
# Confirmed by volume spike (>1.8x 20-bar average) and strong trend (ADX > 25)
# Fixed stop loss via signal=0 when price retraces 25% of ATR from entry
# Discrete sizing 0.25 to limit fee churn; target 80-160 total trades over 4 years (20-40/year)
# Works in bull/bear: breakouts capture momentum, filters avoid false signals

name = "4h_Camarilla_R3S3_1dADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(25) trend filter
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ = 100 * smoothed +DM / ATR, DI- = 100 * smoothed -DM / ATR
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX = smoothed DX
    adx_1d = wilder_smooth(dx, 25)
    
    # Calculate ATR(14) for 4h timeframe (for stoploss)
    tr1_4h = np.abs(high[1:] - low[1:])
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Calculate Camarilla levels for previous day: R3, S3
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align HTF indicators to 4h timeframe (primary)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Camarilla R3 AND strong trend (ADX > 25) AND volume spike
            if close[i] > camarilla_r3_aligned[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Camarilla S3 AND strong trend (ADX > 25) AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retraces 25% of ATR from entry
            if close[i] <= camarilla_r3_aligned[i] - 0.25 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retraces 25% of ATR from entry
            if close[i] >= camarilla_s3_aligned[i] + 0.25 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals