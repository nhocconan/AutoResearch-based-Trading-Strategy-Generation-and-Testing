#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot level reversal with 1d ADX trend filter and volume confirmation.
Long at S1 when price rejects with bullish candle and strong ADX trend, short at R1 with bearish candle.
Exit at midpoint or when ADX weakens. Designed for low trade frequency (20-40/year) to minimize fee drag.
Works in both bull and bear markets by using ADX to filter strong trends and Camarilla levels for mean reversion.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load daily data for ADX filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate 45-period Camarilla levels from previous day
    # H(L) = high, L(L) = low, C(L) = close of previous day
    phigh = df_daily['high'].shift(1).values
    plow = df_daily['low'].shift(1).values
    pclose = df_daily['close'].shift(1).values
    
    # Camarilla levels: H3, H4, L3, L4
    # H4 = C + 1.5 * (H - L), H3 = C + 1.125 * (H - L)
    # L3 = C - 1.125 * (H - L), L4 = C - 1.5 * (H - L)
    camarilla_H4 = pclose + 1.5 * (phigh - plow)
    camarilla_H3 = pclose + 1.125 * (phigh - plow)
    camarilla_L3 = pclose - 1.125 * (phigh - plow)
    camarilla_L4 = pclose - 1.5 * (phigh - plow)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_H4)
    H3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_H3)
    L3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_L3)
    L4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_L4)
    
    # Calculate 1d ADX (14-period)
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # True Range
    tr1 = high_d - low_d
    tr2 = abs(high_d - close_d.shift(1))
    tr3 = abs(low_d - close_d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_d.diff()
    down_move = -low_d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_d)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_d = dx.rolling(window=14, min_periods=14).mean()
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_d.values)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to access previous candle
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish candle: close > open
            bullish = close[i] > open_price[i]
            # Bearish candle: close < open
            bearish = close[i] < open_price[i]
            
            # Long: Price rejects at L3/L4 with bullish candle, strong ADX, volume
            if (bullish and 
                (close[i] <= L3_aligned[i] * 1.001 or close[i] <= L4_aligned[i] * 1.001) and
                adx_aligned[i] > 25 and  # Strong trend
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price rejects at H3/H4 with bearish candle, strong ADX, volume
            elif (bearish and 
                  (close[i] >= H3_aligned[i] * 0.999 or close[i] >= H4_aligned[i] * 0.999) and
                  adx_aligned[i] > 25 and  # Strong trend
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches midpoint (H3/L3 average) OR ADX weakens
                midpoint = (H3_aligned[i] + L3_aligned[i]) / 2.0
                if close[i] >= midpoint or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches midpoint OR ADX weakens
                midpoint = (H3_aligned[i] + L3_aligned[i]) / 2.0
                if close[i] <= midpoint or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_CamarillaReversal_1dADX_Volume"
timeframe = "4h"
leverage = 1.0
#%%