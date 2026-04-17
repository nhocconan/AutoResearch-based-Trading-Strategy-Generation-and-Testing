#!/usr/bin/env python3
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
    
    # Get 1d data for ATR and CCI calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily CCI (20)
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_tp_1d) / (0.015 * mad_1d)
    
    # Align daily indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # 6h ATR for stop calculation
    tr_6h1 = high - low
    tr_6h2 = np.abs(high - np.roll(close, 1))
    tr_6h3 = np.abs(low - np.roll(close, 1))
    tr_6h1[0] = np.nan
    tr_6h2[0] = np.nan
    tr_6h3[0] = np.nan
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 24-period average (6h * 4 = 24h)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 24  # Need volume MA24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma24[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(cci_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period average
        volume_filter = volume[i] > (2.0 * volume_ma24[i])
        # CCI filter: CCI > 100 for long, CCI < -100 for short (strong trend)
        cci_filter = cci_1d_aligned[i] > 100 or cci_1d_aligned[i] < -100
        
        if position == 0:
            # Long: CCI > 100 with volume confirmation
            if cci_1d_aligned[i] > 100 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: CCI < -100 with volume confirmation
            elif cci_1d_aligned[i] < -100 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CCI drops below 0 or volatility drops
            if cci_1d_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CCI rises above 0 or volatility drops
            if cci_1d_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_CCI100_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0