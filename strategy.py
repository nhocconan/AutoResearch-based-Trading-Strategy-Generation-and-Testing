#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_VolumeSpike_ATRFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Donchian Channels (20-period) from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values for Donchian calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # 20-period high and low from previous day's data
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 4h ATR for volatility filter and stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume Spike Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_ratio_val = vol_ratio[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(upper) or 
            np.isnan(lower) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume spike
            if close_val > upper and vol_ratio_val > 2.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Break below lower Donchian with volume spike
            elif close_val < lower and vol_ratio_val > 2.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long: Exit on stoploss or reversal signal
            if low_val <= entry_price - 2.0 * atr_val or close_val < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: Exit on stoploss or reversal signal
            if high_val >= entry_price + 2.0 * atr_val or close_val > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals