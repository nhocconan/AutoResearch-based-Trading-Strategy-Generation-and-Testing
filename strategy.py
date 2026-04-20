#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1h and 4h data for analysis
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h Bollinger Bands (20, 2)
    close_1h = df_1h['close'].values
    sma20 = pd.Series(close_1h).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1h).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    upper_band_aligned = align_htf_to_ltf(prices, df_1h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1h, lower_band)
    
    # 1h volume spike (20-period)
    volume_1h = df_1h['volume'].values
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        vol = volume_1h[i]
        
        if position == 0:
            # Long: price above upper BB, above 4h EMA50, with volume confirmation
            if (price > upper_band_aligned[i] and 
                price > ema50_4h_aligned[i] and 
                vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lower BB, below 4h EMA50, with volume confirmation
            elif (price < lower_band_aligned[i] and 
                  price < ema50_4h_aligned[i] and 
                  vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below lower BB or trend reversal
            if (price < lower_band_aligned[i] or 
                price < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above upper BB or trend reversal
            if (price > upper_band_aligned[i] or 
                price > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1h_4h_BB_Trend_Volume"
timeframe = "1h"
leverage = 1.0