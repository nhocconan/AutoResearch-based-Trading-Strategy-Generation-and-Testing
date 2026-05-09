#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_EquiVolume_Wave_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EquiVolume Wave calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EquiVolume Wave: (Close - Low) / (High - Low) * Volume
    # This measures buying pressure (close near high with volume)
    hl_range = df_1d['high'] - df_1d['low']
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # Avoid division by zero
    ev_wave = ((df_1d['close'] - df_1d['low']) / hl_range) * df_1d['volume']
    
    # Smooth the wave with a 5-period SMA
    ev_wave_smooth = pd.Series(ev_wave.values).rolling(window=5, min_periods=5).mean().values
    
    # Trend filter: 1d EMA21
    ema21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align both to 4h
    ev_wave_4h = align_htf_to_ltf(prices, df_1d, ev_wave_smooth)
    ema21_1d_4h = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for smoothing
    
    for i in range(start_idx, n):
        if (np.isnan(ev_wave_4h[i]) or np.isnan(ema21_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wave_val = ev_wave_4h[i]
        trend = ema21_1d_4h[i]
        
        if position == 0:
            # Enter long: strong buying pressure (high wave) and above trend
            if wave_val > 0 and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: weak buying pressure (low/negative wave) and below trend
            elif wave_val < 0 and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: buying pressure fades (wave turns negative)
            if wave_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: buying pressure returns (wave turns positive)
            if wave_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals