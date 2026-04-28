#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation.
# Uses proven Camarilla structure from DB top performers. Tight entries target 12-37 trades/year.
# Long when price breaks above R3 with volume and 1d ADX > 25 (strong trend).
# Short when price breaks below S3 with volume and 1d ADX > 25 (strong trend).
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.

name = "6h_Camarilla_R3_S3_Breakout_1dADX25_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original indices
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        tr_smoothed = WilderSmoothing(tr, period)
        dm_plus_smoothed = WilderSmoothing(dm_plus, period)
        dm_minus_smoothed = WilderSmoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smoothed / np.where(tr_smoothed == 0, np.nan, tr_smoothed)
        di_minus = 100 * dm_minus_smoothed / np.where(tr_smoothed == 0, np.nan, tr_smoothed)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
        adx = WilderSmoothing(dx, period)
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate 6h Camarilla levels (based on previous bar's range)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close + (high - low) * 1.1 / 2
    camarilla_s3 = close - (high - low) * 1.1 / 2
    # Shift by 1 to use previous bar's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient history for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_14_1d_aligned[i] > 25
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r3[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s3[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s3[i]
        short_exit = close[i] > camarilla_r3[i]
        
        # Handle entries and exits
        if long_breakout and strong_trend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and strong_trend and position >= 0:
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