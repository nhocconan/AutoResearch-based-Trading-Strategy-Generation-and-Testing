#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_VWAP_Breakout_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP using typical price
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(vwap_numerator, np.nan), 
                        where=vwap_denominator!=0)
    
    # Calculate 1-day ATR(14) for volatility-based bands
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate dynamic deviation multiplier based on ATR percentile (adaptive bands)
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=30, min_periods=10).rank(pct=True).values
    # Scale deviation: low vol -> tighter bands (0.5), high vol -> wider bands (2.0)
    dev_multiplier = 0.5 + 1.5 * atr_percentile  # ranges from 0.5 to 2.0
    
    # Calculate upper and lower VWAP bands using previous day's VWAP
    prev_vwap = np.concatenate([[np.nan], vwap_1d[:-1]])
    atr_prev = np.concatenate([[np.nan], atr_14[:-1]])
    dev_prev = np.concatenate([[np.nan], dev_multiplier[:-1]])
    
    upper_band = prev_vwap + atr_prev * dev_prev
    lower_band = prev_vwap - atr_prev * dev_prev
    
    # Align VWAP bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume filter: current 12h volume > 1.5x 20-period average (approx 10-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        if np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper VWAP band with volume
            if price > upper_band_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower VWAP band with volume
            elif price < lower_band_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below upper band
            if price < upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above lower band
            if price > lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals