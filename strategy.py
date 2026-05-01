#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d ADX trend filter and volume confirmation
# Uses 1d ADX > 25 to identify strong trends (both bull and bear)
# Williams %R(14) identifies oversold/overbought conditions within the trend
# Long when %R < -80 (oversold) in bullish trend (ADX > 25 + +DI > -DI)
# Short when %R > -20 (overbought) in bearish trend (ADX > 25 + -DI > +DI)
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-30 trades/year per symbol with 0.25 sizing
# ADX filter reduces false signals in ranging markets while capturing strong trends
# Works in both bull and bear markets by following the dominant daily trend direction

name = "12h_WilliamsR_1dADX_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smoothed = WilderSmoothing(tr, period)
    dm_plus_smoothed = WilderSmoothing(dm_plus, period)
    dm_minus_smoothed = WilderSmoothing(dm_minus, period)
    
    # DI and DX
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = WilderSmoothing(dx, period)
    
    # Align HTF indicators to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # Williams %R calculation (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d data for ADX (14+14+14=42 days) + Williams %R (14) + volume EMA20
    start_idx = max(42, 14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or 
            np.isnan(di_minus_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX: need ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        bullish_trend = strong_trend and (di_plus_aligned[i] > di_minus_aligned[i])
        bearish_trend = strong_trend and (di_minus_aligned[i] > di_plus_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            if bullish_trend:
                # Long: Williams %R oversold (< -80) in bullish trend with volume spike
                if williams_r[i] < -80 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_trend:
                # Short: Williams %R overbought (> -20) in bearish trend with volume spike
                if williams_r[i] > -20 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No strong trend or wrong DI alignment
        
        elif position == 1:  # Long position
            # Exit: Williams %R reaches overbought (> -20) or trend weakens (ADX < 20)
            if williams_r[i] > -20 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R reaches oversold (< -80) or trend weakens (ADX < 20)
            if williams_r[i] < -80 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals