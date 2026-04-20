#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d CCI(20) - identifies overbought/oversold conditions
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp_1d - sma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (tp_1d - sma_tp) / (0.015 * mad)
    cci[mad == 0] = 0
    cci_cci = cci  # for clarity
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_cci)
    
    # 1d ATR(14) for volatility filter and position sizing
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 1d Volume ratio (current volume / 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Load 12h data for price action (prices dataframe is already at 12h timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(cci_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # CCI thresholds: oversold < -100, overbought > 100
        cci_val = cci_aligned[i]
        
        # Volatility filter: only trade when volatility is moderate
        atr = atr_14_aligned[i]
        atr_ma_20 = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 2.0 * atr_ma_20)
        
        # Volume filter: require above-average volume for confirmation
        vol_filter = vol_filter and (vol_ratio_aligned[i] > 1.2)
        
        if position == 0:
            # Long when CCI indicates oversold conditions with volume confirmation
            if (cci_val < -100) and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when CCI indicates overbought conditions with volume confirmation
            elif (cci_val > 100) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI returns to neutral territory or volatility spikes
            if (cci_val > 0) or (atr > 3.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI returns to neutral territory or volatility spikes
            if (cci_val < 0) or (atr > 3.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_CCI20_OverboughtOversold_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0