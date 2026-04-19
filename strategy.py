#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_TrailingStop_Minimal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_10_1d = pd.Series(tr1).rolling(window=10, min_periods=10).mean().values
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Daily ATR for stop loss
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr_10_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_vol = atr_10_1d_aligned[i]
        atr_stop = atr_14_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        low_vol_regime = atr_vol < np.nanmedian(atr_10_1d_aligned[:i+1])  # Below median ATR = low volatility
        
        if position == 0:
            # Enter long on upper breakout with volume confirmation in low volatility
            if price > upper[i] and volume_confirmed and low_vol_regime:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Enter short on lower breakdown with volume confirmation in low volatility
            elif price < lower[i] and volume_confirmed and low_vol_regime:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        
        elif position == 1:
            # Track highest price since entry for trailing stop
            highest_since_entry = max(highest_since_entry, price)
            # Exit if price drops 2.5x ATR from high or breaks below lower band
            if price < highest_since_entry - 2.5 * atr_stop or price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Track lowest price since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, price)
            # Exit if price rises 2.5x ATR from low or breaks above upper band
            if price > lowest_since_entry + 2.5 * atr_stop or price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals