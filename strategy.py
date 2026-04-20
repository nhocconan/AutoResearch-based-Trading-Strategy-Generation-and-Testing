#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Volume_Spike_Trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === Daily ATR for volatility regime (14-day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Volume Spike (20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h Trend Filter: 50-period EMA ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_spike_val = vol_spike[i]
        ema50_val = ema50[i]
        atr14_val = atr14[i]
        
        # Skip if any value is NaN
        if (np.isnan(donch_high_val) or np.isnan(donch_low_val) or 
            np.isnan(vol_spike_val) or np.isnan(ema50_val) or np.isnan(atr14_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and uptrend
            if (close_val > donch_high_val and 
                vol_spike_val > 2.0 and
                close_val > ema50_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and downtrend
            elif (close_val < donch_low_val and 
                  vol_spike_val > 2.0 and
                  close_val < ema50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low or trend breaks
            if close_val < donch_low_val or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high or trend breaks
            if close_val > donch_high_val or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals