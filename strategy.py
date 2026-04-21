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
    
    # === Daily Donchian Channels (20) for breakout ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === Daily ATR (14) for volatility filter and stoploss ===
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(df_1d['close']).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(df_1d['close']).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price_close = prices['close'].iloc[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        atr = atr_14_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long on Donchian breakout with volume
            if price_close > upper_band and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Enter short on Donchian breakdown with volume
            elif price_close < lower_band and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2 * ATR from entry
            if position == 1 and price_close < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif position == -1 and price_close > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit on opposite Donchian touch
            elif position == 1 and price_close < lower_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif position == -1 and price_close > upper_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_ATR_Volume_Filter_v2"
timeframe = "4h"
leverage = 1.0