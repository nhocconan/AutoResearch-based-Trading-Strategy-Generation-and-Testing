#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level and close > 1d EMA34 (uptrend) with volume > 2.0x average.
Short when price breaks below Camarilla S3 level and close < 1d EMA34 (downtrend) with volume > 2.0x average.
Exit on opposite Camarilla level break or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Camarilla levels provide intraday support/resistance, EMA34 filters medium-term trend, volume spike confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in choppy markets across both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar (avoid look-ahead)
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    high_low_4h = high_4h - low_4h
    r3_4h = typical_price_4h + (high_low_4h * 1.1 / 2.0)
    s3_4h = typical_price_4h - (high_low_4h * 1.1 / 2.0)
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r3_val and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s3_val and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR trend reversal
                if (price < s3_val or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR trend reversal
                if (price > r3_val or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0