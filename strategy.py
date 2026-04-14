#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily True Range and ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily ATR-based volatility filter (low volatility regime)
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    low_volatility = atr_14_1d < 0.8 * atr_ma_50_1d  # Below average volatility
    
    # Daily Bollinger Bands (20, 2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Calculate median volume for volume spike filter
    vol_median = np.nanmedian(volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Get aligned daily indicators
        atr_14_1d_i = align_htf_to_ltf(prices, df_1d, atr_14_1d)[i]
        atr_ma_50_1d_i = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)[i]
        low_vol_i = align_htf_to_ltf(prices, df_1d, low_volatility)[i]
        sma_20_1d_i = align_htf_to_ltf(prices, df_1d, sma_20_1d)[i]
        std_20_1d_i = align_htf_to_ltf(prices, df_1d, std_20_1d)[i]
        upper_bb_1d_i = align_htf_to_ltf(prices, df_1d, upper_bb_1d)[i]
        lower_bb_1d_i = align_htf_to_ltf(prices, df_1d, lower_bb_1d)[i]
        
        if np.isnan(atr_14_1d_i) or np.isnan(atr_ma_50_1d_i) or np.isnan(low_vol_i) or \
           np.isnan(sma_20_1d_i) or np.isnan(std_20_1d_i) or np.isnan(upper_bb_1d_i) or np.isnan(lower_bb_1d_i):
            continue
        
        # Low volatility regime filter
        if not low_vol_i:
            continue
        
        # Volume spike filter
        volume_spike = volume[i] > 1.5 * vol_median
        
        # Bollinger Band squeeze conditions
        bb_width = (upper_bb_1d_i - lower_bb_1d_i) / sma_20_1d_i
        bb_width_ma = pd.Series(((upper_bb_1d - lower_bb_1d) / sma_20_1d)).rolling(window=20, min_periods=20).mean().values
        bb_width_ma_i = align_htf_to_ltf(prices, df_1d, bb_width_ma)[i]
        squeeze = bb_width < 0.7 * bb_width_ma_i  # Bollinger Band squeeze
        
        # Long conditions: price breaks above upper BB with volume in low vol squeeze
        if position == 0 and volume_spike and squeeze:
            if close[i] > upper_bb_1d_i:
                position = 1
                signals[i] = position_size
            # Short conditions: price breaks below lower BB with volume in low vol squeeze
            elif close[i] < lower_bb_1d_i:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions: price returns to middle Bollinger Band (mean reversion)
        elif position == 1:
            if close[i] < sma_20_1d_i:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            if close[i] > sma_20_1d_i:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Bollinger_Squeeze_Volume_Breakout"
timeframe = "12h"
leverage = 1.0