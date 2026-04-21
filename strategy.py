#!/usr/bin/env python3
"""
6h_HTF_12h_Camarilla_R1S1_Breakout_VolumeFilter_V1
Hypothesis: 6h Camarilla R1/S1 breakout with 12h volume confirmation and 12h EMA34 trend filter. 
Camarilla levels from 12h provide institutional support/resistance. Volume >1.5x 20-period MA confirms breakout strength. 
EMA34 on 12h filters for higher-timeframe trend alignment. Works in bull/bear by trading with 12h trend only.
Target: 12-37 trades/year (50-150 total over 4 years) using discrete 0.25 position sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for Camarilla, volume, EMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h Camarilla Levels (using previous 12h bar's OHLC) ===
    # Camarilla calculation: based on previous bar's range
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar (based on previous bar)
    R1 = np.full(len(df_12h), np.nan)
    S1 = np.full(len(df_12h), np.nan)
    R3 = np.full(len(df_12h), np.nan)
    S3 = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        # Previous bar's OHLC
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        range_ = prev_high - prev_low
        
        # Camarilla levels
        R1[i] = prev_close + range_ * 1.1 / 12
        S1[i] = prev_close - range_ * 1.1 / 12
        R3[i] = prev_close + range_ * 1.1 / 4
        S3[i] = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (no extra delay - levels known at bar open)
    R1_6h = align_htf_to_ltf(prices, df_12h, R1)
    S1_6h = align_htf_to_ltf(prices, df_12h, S1)
    R3_6h = align_htf_to_ltf(prices, df_12h, R3)
    S3_6h = align_htf_to_ltf(prices, df_12h, S3)
    
    # === 12h EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h Volume MA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 6h primary data ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i])
            or np.isnan(ema_34_6h[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma_6h[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + 12h uptrend (price > EMA34)
            if price > R1_6h[i] and vol_ok and price > ema_34_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + 12h downtrend (price < EMA34)
            elif price < S1_6h[i] and vol_ok and price < ema_34_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or volume confirmation fails
            if price < S1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or volume confirmation fails
            if price > R1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_12h_Camarilla_R1S1_Breakout_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0