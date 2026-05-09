#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when: close > Camarilla R3, 1d EMA34 rising, volume spike (>1.5x 20-period average)
# Short when: close < Camarilla S3, 1d EMA34 falling, volume spike
# Exit when: price crosses Camarilla H/L OR trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 25-50 trades/year.
# Uses proven Camarilla structure with EMA trend filter and volume confirmation for edge in both bull and bear markets.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior day (high, low, close)
    # Camarilla: H/L = (H-L) * 1.1/2 + C; R3/S3 = C ± (H-L) * 1.1/2 * 1.5
    # Using prior day's OHLC to avoid look-ahead
    daily_high = pd.Series(high).rolling(window=96, min_periods=96).max()  # 96*4h = 16d, but we want daily
    daily_low = pd.Series(low).rolling(window=96, min_periods=96).min()
    daily_close = pd.Series(close).rolling(window=96, min_periods=96).last()
    
    # Actually, better to get true daily data from 1d timeframe
    # We'll use 1d data for Camarilla calculation
    
    # Get 1d data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Camarilla levels: R3, S3, H/L using 1d OHLC
    # R3 = Close + (High-Low) * 1.1/2 * 1.5
    # S3 = Close - (High-Low) * 1.1/2 * 1.5
    # H/L = (High+Low)/2 (pivot)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rang = high_1d - low_1d
    camarilla_h_l = (high_1d + low_1d) / 2
    camarilla_r3 = close_1d + rang * 1.1 / 2 * 1.5
    camarilla_s3 = close_1d - rang * 1.1 / 2 * 1.5
    
    # Align Camarilla levels to 4h
    camarilla_h_l_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h_l)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h_l_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + 1d EMA34 rising + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Camarilla S3 + 1d EMA34 falling + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla H/L OR trend turns down
            if (close[i] < camarilla_h_l_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla H/L OR trend turns up
            if (close[i] > camarilla_h_l_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals