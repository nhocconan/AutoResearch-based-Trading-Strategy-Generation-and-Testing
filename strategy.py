#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (using prior day's OHLC)
    # We shift by 1 to ensure we only use completed daily bars
    phigh_1d = np.roll(high_1d, 1)
    plow_1d = np.roll(low_1d, 1)
    pclose_1d = np.roll(close_1d, 1)
    
    # Set first value to NaN since there's no previous day
    phigh_1d[0] = np.nan
    plow_1d[0] = np.nan
    pclose_1d[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    camarilla_r1 = pclose_1d + (phigh_1d - plow_1d) * 1.1 / 12
    camarilla_s1 = pclose_1d - (phigh_1d - plow_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume spike for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_12h[i]) or np.isnan(camarilla_s1_12h[i]) or 
            np.isnan(ema34_1d_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + above daily EMA34 + volume spike
            if (close[i] > camarilla_r1_12h[i] and 
                close[i] > ema34_1d_12h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + below daily EMA34 + volume spike
            elif (close[i] < camarilla_s1_12h[i] and 
                  close[i] < ema34_1d_12h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR below daily EMA34
            if close[i] < camarilla_s1_12h[i] or close[i] < ema34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR above daily EMA34
            if close[i] > camarilla_r1_12h[i] or close[i] > ema34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals