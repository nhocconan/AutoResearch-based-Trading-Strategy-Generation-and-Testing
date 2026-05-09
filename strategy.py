#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX (15-period) on 1d close
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = np.where(ema2[:-1] != 0, (ema3[1:] - ema2[:-1]) / ema2[:-1] * 100, 0)
    trix_raw = np.concatenate([[0], trix_raw])  # align length
    
    # Align TRIX to 4h
    trix = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Chop filter on 1d (14-period)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[0], tr2])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1d * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) != 0, chop, 50)
    chop = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike filter: current 4h volume > 2.0 * 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need enough data for TRIX and chop
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        t = trix[i]
        c = chop[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: TRIX crosses above -0.1, chop > 61.8 (range), volume spike
            if t > -0.1 and trix[i-1] <= -0.1 and c > 61.8 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below +0.1, chop > 61.8 (range), volume spike
            elif t < 0.1 and trix[i-1] >= 0.1 and c > 61.8 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop < 38.2 (trend)
            if t < 0 and trix[i-1] >= 0 or c < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop < 38.2 (trend)
            if t > 0 and trix[i-1] <= 0 or c < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals