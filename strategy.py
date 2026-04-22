#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Camarilla pivots derived from previous day's range provide strong support/resistance levels.
# Breakout above R3 or below S3 with volume confirmation indicates institutional interest.
# 1d EMA34 filter ensures trades align with higher timeframe trend.
# Designed to work in both bull and bear markets by requiring trend alignment.
# Targets 25-40 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R3 = Close + 1.1*(High-Low)*1.1/2, S3 = Close - 1.1*(High-Low)*1.1/2
    # Actually: R3 = Close + 1.1*(High-Low), S3 = Close - 1.1*(High-Low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_1d_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above R3 + uptrend + volume spike
            if price > r3_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below S3 + downtrend + volume spike
            elif price < s3_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns below R3 or trend breaks
                if price < r3_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns above S3 or trend breaks
                if price > s3_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0