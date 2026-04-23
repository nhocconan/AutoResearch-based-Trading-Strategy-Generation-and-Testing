#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
- Uses Camarilla pivot levels (R3/S3) from 1d for high-probability breakout zones
- Volume confirmation (> 1.5x 20-period average) ensures momentum behind breakouts
- ADX(14) > 25 on 1d filters for trending markets, avoids whipsaws in chop
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Tight entry conditions minimize fee drag while capturing strong moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's range
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d ADX(14) for trend filter
    def calculate_atr(high, low, close, period):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        result = np.full_like(tr, np.nan)
        if len(tr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(tr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(tr)):
            result[i] = (result[i-1] * (period-1) + tr[i]) / period
        return result
    
    atr = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # +DI and -DI calculation
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the DM values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    smoothed_plus_dm = wilders_smoothing(plus_dm, 14)
    smoothed_minus_dm = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / atr
    minus_di = 100 * smoothed_minus_dm / atr
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Camarilla needs 1 day buffer, ADX needs warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_r3 = close[i] > camarilla_r3_aligned[i]
        price_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above R3, strong trend, volume spike
            long_signal = (price_above_r3 and 
                          strong_trend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S3, strong trend, volume spike
            short_signal = (price_below_s3 and 
                           strong_trend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend weakening
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 or trend weakens
                if (price_below_s3 or 
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or trend weakens
                if (price_above_r3 or 
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dADXTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0