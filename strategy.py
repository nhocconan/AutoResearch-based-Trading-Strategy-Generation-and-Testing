#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Donchian channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper/lower bands
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Donchian middle (midpoint)
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2
    
    # Align to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_20)
    
    # === Daily volume confirmation (20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h price data ===
    close_6h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_6h[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        donch_mid = donch_mid_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        if position == 0:
            # Breakout above upper band with volume confirmation
            if price_close > donch_high and vol_ratio_val > 1.3:
                signals[i] = 0.25
                position = 1
            # Breakdown below lower band with volume confirmation
            elif price_close < donch_low and vol_ratio_val > 1.3:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to middle (mean reversion)
            if position == 1 and price_close < donch_mid:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_Breakout_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0