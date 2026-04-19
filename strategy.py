#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_VWAP_Deviation_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP (volume-weighted average price)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_num = np.cumsum(typical_price_1w * volume_1w)
    vwap_den = np.cumsum(volume_1w)
    vwap_1w = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    # Calculate weekly VWAP standard deviation
    squared_dev = (typical_price_1w - vwap_1w) ** 2 * volume_1w
    var_num = np.cumsum(squared_dev)
    vwap_var = np.where(vwap_den != 0, var_num / vwap_den, np.nan)
    vwap_std_1w = np.sqrt(np.maximum(vwap_var, 0))
    
    # Upper and lower bands (2 standard deviations)
    upper_band_1w = vwap_1w + 2.0 * vwap_std_1w
    lower_band_1w = vwap_1w - 2.0 * vwap_std_1w
    
    # Align weekly VWAP bands to 12h timeframe
    vwap_12h = align_htf_to_ltf(prices, df_1w, vwap_1w)
    upper_12h = align_htf_to_ltf(prices, df_1w, upper_band_1w)
    lower_12h = align_htf_to_ltf(prices, df_1w, lower_band_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(vwap_12h[i]) or np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above upper VWAP band with volume
            if price > upper_12h[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower VWAP band with volume
            elif price < lower_12h[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below VWAP
            if price < vwap_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above VWAP
            if price > vwap_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals