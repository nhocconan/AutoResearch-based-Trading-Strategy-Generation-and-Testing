#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation. Uses 1h Williams %R for precise entry timing. Designed for BTC/ETH robustness: Camarilla breakouts capture institutional order flow, EMA34 filter ensures alignment with higher timeframe trend, volume spike confirms participation, and Williams %R on 1h provides mean-reversion entry within the breakout direction. Targets 20-50 trades/year on 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_1hWMR_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1h Williams %R (14-period) for entry timing
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    highest_high_1h = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low_1h = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r_1h = -100 * (highest_high_1h - close_1h) / (highest_high_1h - lowest_low_1h)
    williams_r_1h = np.where((highest_high_1h - lowest_low_1h) == 0, -50, williams_r_1h)
    williams_r_1h_aligned = align_htf_to_ltf(prices, df_1h, williams_r_1h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels (using prior day's close, high, low)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_1h_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above Camarilla R3, price > 1d EMA34, volume spike (>2.0x avg), Williams %R not overbought (> -80)
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i] and 
                williams_r_1h_aligned[i] > -80):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S3, price < 1d EMA34, volume spike (>2.0x avg), Williams %R not oversold (< -20)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i] and 
                  williams_r_1h_aligned[i] < -20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla S3 (mean reversion) OR Williams %R becomes oversold (< -80)
            if (close[i] <= camarilla_s3_aligned[i] or 
                williams_r_1h_aligned[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla R3 (mean reversion) OR Williams %R becomes overbought (> -20)
            if (close[i] >= camarilla_r3_aligned[i] or 
                williams_r_1h_aligned[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals