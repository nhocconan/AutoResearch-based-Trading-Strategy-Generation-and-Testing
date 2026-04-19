#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Supertrend_TrendFilter_VolumeBreakout"
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
    
    # Get 12h data for Supertrend (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend (12h)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_period = 10
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_12h, np.nan)
    uptrend = np.full_like(close_12h, True)
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1] if i > 0 else True
            continue
            
        if close_12h[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close_12h[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # Align Supertrend and trend direction to 4h timeframe
    supertrend_4h = align_htf_to_ltf(prices, df_12h, supertrend)
    uptrend_4h = align_htf_to_ltf(prices, df_12h, uptrend.astype(float))  # 1.0 for uptrend, 0.0 for downtrend
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_4h[i]) or np.isnan(uptrend_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above Supertrend in uptrend with volume
            if price > supertrend_4h[i] and uptrend_4h[i] > 0.5 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Supertrend in downtrend with volume
            elif price < supertrend_4h[i] and uptrend_4h[i] < 0.5 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below Supertrend
            if price < supertrend_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above Supertrend
            if price > supertrend_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals