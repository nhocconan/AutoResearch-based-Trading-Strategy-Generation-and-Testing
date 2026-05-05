#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX(14) trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 level AND 4h ADX > 25 AND volume > 1.5x 20-period average
# Short when price breaks below 1h Camarilla S3 level AND 4h ADX > 25 AND volume > 1.5x 20-period average
# Exit when price crosses 1h Camarilla pivot point (mean reversion) OR 4h ADX < 20 (trend weakening)
# Uses 1h primary timeframe with 4h HTF for ADX trend filter
# Camarilla levels provide clear breakout zones based on previous hour's range
# ADX filter ensures we only trade in trending markets, reducing whipsaw in ranges
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_Camarilla_R3S3_Breakout_4hADX_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h ADX(14) for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing function
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1h data ONCE before loop for Camarilla levels (based on previous hour)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous hour's OHLC
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    open_1h = df_1h['open'].values
    
    # Camarilla R3 and S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1h + (1.1 * (high_1h - low_1h) / 2)
    camarilla_s3 = close_1h - (1.1 * (high_1h - low_1h) / 2)
    camarilla_pivot = (high_1h + low_1h + close_1h) / 3  # Standard pivot point
    
    # Align to 1h timeframe (using previous hour's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pivot)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND ADX > 25 AND volume spike AND session
            if (close[i] > camarilla_r3_aligned[i] and 
                adx_4h_aligned[i] > 25 and 
                volume_filter[i] and
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND ADX > 25 AND volume spike AND session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  adx_4h_aligned[i] > 25 and 
                  volume_filter[i] and
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] < camarilla_pivot_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] > camarilla_pivot_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals