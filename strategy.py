#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d ADX regime filter + volume confirmation.
# Long when: price > Alligator Jaw (teeth > lips) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA.
# Short when: price < Alligator Jaw (teeth < lips) AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
# Exit when: price crosses Alligator Jaw OR 1d ADX < 20 (range regime).
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_WilliamsAlligator_1dADX25_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest is Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    # ADX is Wilder's smoothing of DX
    if len(dx) >= period:
        adx[period-1] = np.nanmean(dx[:period])
        for i in range(period, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator (using 5m/6h equivalent periods: 13,8,5 but scaled to 6h)
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(data, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest is SMMA: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply Alligator shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        # Alligator conditions: Jaw is the reference line
        # Long bias: teeth > lips (green alignment)
        # Short bias: teeth < lips (red alignment)
        alligator_long = teeth[i] > lips[i]
        alligator_short = teeth[i] < lips[i]
        
        # Regime filter: 1d ADX > 25 for trending market
        strong_trend = adx_aligned[i] > 25
        
        # Exit regime: ADX < 20 suggests ranging market
        ranging_market = adx_aligned[i] < 20
        
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price > Jaw AND alligator long alignment AND strong trend AND volume spike
            if close_val > jaw[i] and alligator_long and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND alligator short alignment AND strong trend AND volume spike
            elif close_val < jaw[i] and alligator_short and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < Jaw OR alligator alignment changes OR ranging market
            if close_val < jaw[i] or not alligator_long or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > Jaw OR alligator alignment changes OR ranging market
            if close_val > jaw[i] or not alligator_short or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals