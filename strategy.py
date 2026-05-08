#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Alligator_Trend"
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
    
    # Get daily data for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / np.where(atr > 0, atr, 1)
    minus_di = 100 * wilders_smooth(minus_dm, 14) / np.where(atr > 0, atr, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = wilders_smooth(dx, 14)
    
    # Alligator (Williams Alligator: Jaw=13, Teeth=8, Lips=5)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align all indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation - 24-period average volume (4 days for 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment check
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: ADX > 25 + +DI > -DI + Bullish Alligator + Volume
            if (adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i] and
                bullish_align and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 + -DI > +DI + Bearish Alligator + Volume
            elif (adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i] and
                  bearish_align and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 OR Bearish Alligator alignment
            if (adx_aligned[i] < 20 or not bullish_align):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 OR Bullish Alligator alignment
            if (adx_aligned[i] < 20 or not bearish_align):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals