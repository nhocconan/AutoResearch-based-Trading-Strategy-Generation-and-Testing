#!/usr/bin/env python3
"""
12h Donchian breakout with 1d ATR and volume filter.
Long when price breaks above 20-period high with volume>1.3x and ATR>0.5% of price.
Short when price breaks below 20-period low with volume>1.3x and ATR>0.5% of price.
Exit on opposite band cross or 2x ATR stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(20)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_1d[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper_band = donch_high[i]
        lower_band = donch_low[i]
        atr_val = atr_1d[i]
        vol_ratio_val = vol_ratio[i]
        price_level = price_close
        
        # ATR filter: only trade when ATR > 0.5% of price
        atr_filter = atr_val > 0.005 * price_level
        
        if position == 0:
            # Enter long: break above upper band with volume and ATR filter
            if (price_high > upper_band and 
                vol_ratio_val > 1.3 and 
                atr_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band with volume and ATR filter
            elif (price_low < lower_band and 
                  vol_ratio_val > 1.3 and 
                  atr_filter):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite band cross OR 2x ATR stop
            exit_signal = False
            
            # Opposite band exit
            if position == 1 and price_close < lower_band:
                exit_signal = True
            elif position == -1 and price_close > upper_band:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry band)
            if position == 1:
                if price_close < upper_band - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if price_close > lower_band + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Volume1.3x_ATR2x"
timeframe = "12h"
leverage = 1.0