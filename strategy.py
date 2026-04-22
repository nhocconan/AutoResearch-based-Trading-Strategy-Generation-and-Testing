#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d trend filter and volume confirmation.
Long when Williams %R crosses above -20 from oversold (<-80) with bullish 1d trend and volume spike.
Short when Williams %R crosses below -80 from overbought (>-20) with bearish 1d trend and volume spike.
Exit when Williams %R returns to -50 or trend reverses.
Uses 1d ADX(14) > 25 for trend strength filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (15-30/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength filter
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def _wilder_smooth(x, period):
        smoothed = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return smoothed
        smoothed[period-1] = np.nansum(x[1:period+1])
        for i in range(period, len(x)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + x[i]
        return smoothed
    
    atr = _wilder_smooth(tr, 14)
    plus_di = 100 * _wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * _wilder_smooth(minus_dm, 14) / atr
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = _wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
        
        # Trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Williams %R crosses above -20 from oversold (<-80) with volume spike
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and
                williams_r[i-1] < -80 and  # Was oversold
                strong_trend and
                volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from overbought (>-20) with volume spike
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and
                  williams_r[i-1] > -20 and  # Was overbought
                  strong_trend and
                  volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to -50 or trend weakens
                if williams_r[i] >= -50 or not strong_trend:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to -50 or trend weakens
                if williams_r[i] <= -50 or not strong_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extreme_ADX25_Volume"
timeframe = "4h"
leverage = 1.0
#%%