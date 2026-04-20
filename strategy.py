#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian Channel (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in HTF indicators
        if np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high_20_aligned[i]
        dl = donchian_low_20_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 50-period average
        vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = vol > vol_ma_50
        
        # Volatility regime: only trade when volatility is below 70th percentile (calm markets)
        vol_regime = atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 70)
        
        if position == 0:
            # Long: price breaks above daily Donchian high with volume confirmation and calm volatility
            if price > dh and vol_filter and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with volume confirmation and calm volatility
            elif price < dl and vol_filter and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily Donchian low or volatility spikes above 85th percentile
            if price < dl or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 85):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily Donchian high or volatility spikes above 85th percentile
            if price > dh or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 85):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeVolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0