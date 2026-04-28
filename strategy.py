#!/usr/bin/env python3
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
    
    # Get daily data for HTF context (Williams Alligator)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (13, 8, 5 SMAs with future shifts)
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    
    # Jaw (13-period, 8-bar shift)
    jaw_1d = close_1d_series.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period, 5-bar shift) 
    teeth_1d = close_1d_series.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period, 3-bar shift)
    lips_1d = close_1d_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # ADX (14-period) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI and DX
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    
    # ADX (smoothed DX)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # ADX filter: trend strength > 25
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions: 
        # Long: bullish Alligator alignment + strong trend + volume
        # Short: bearish Alligator alignment + strong trend + volume
        long_entry = bullish_alligator and strong_trend and vol_filter
        short_entry = bearish_alligator and strong_trend and vol_filter
        
        # Exit conditions: Alligator lines cross (Lips crosses Teeth)
        long_exit = lips_aligned[i] < teeth_aligned[i]
        short_exit = lips_aligned[i] > teeth_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "6h_WilliamsAlligator_ADX25_Volume_Session"
timeframe = "6h"
leverage = 1.0