#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 12h ADX trend filter + volume spike confirmation
# Long when Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weakens)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Williams %R identifies exhaustion points in trending markets, ADX filters for sufficient trend strength,
# volume confirmation ensures institutional participation. Works in bull markets via buying oversold dips
# in uptrends and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_Extreme_12hADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(df_6h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_6h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - df_6h['close'].values) / (highest_high_14 - lowest_low_14) * -100
    # Replace division by zero or near-zero with NaN
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, np.nan, williams_r)
    
    # Align Williams %R to prices timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    # ADX requires +DI, -DI, and TR calculations
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_smoothed = wilders_smoothing(tr, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
    minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Uptrend when ADX > 25, weak trend when ADX < 20
    strong_trend = adx > 25
    weak_trend = adx < 20
    
    # Align 12h ADX indicators to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend.astype(float))
    weak_trend_aligned = align_htf_to_ltf(prices, df_12h, weak_trend.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(strong_trend_aligned[i]) or 
            np.isnan(weak_trend_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND strong trend AND volume spike
            if (williams_r_aligned[i] < -80 and 
                strong_trend_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND strong trend AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR trend weakens (ADX < 20)
            if (williams_r_aligned[i] > -50 or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR trend weakens (ADX < 20)
            if (williams_r_aligned[i] < -50 or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals