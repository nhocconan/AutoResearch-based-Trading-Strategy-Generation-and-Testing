#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (S1/S3 for long, R1/R3 for short) with 1w EMA trend filter and volume confirmation.
# Camarilla pivot levels provide precise support/resistance zones from prior 1d session.
# Long when price touches S1/S3 in uptrend (price > 1w EMA), short when touches R1/R3 in downtrend (price < 1w EMA).
# Volume confirmation requires current volume > 2.0x 20-period average to avoid false breakouts.
# Designed to work in both bull and bear markets by aligning with 1w trend filter.
# Targets 15-30 trades/year with strict entry conditions to minimize fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on 1w data
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load 1d data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: 
    #   S1 = close - (high - low) * 1.05 / 6
    #   S2 = close - (high - low) * 1.10 / 6
    #   S3 = close - (high - low) * 1.15 / 6
    #   R1 = close + (high - low) * 1.05 / 6
    #   R2 = close + (high - low) * 1.10 / 6
    #   R3 = close + (high - low) * 1.15 / 6
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.05 / 6.0
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.15 / 6.0
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.05 / 6.0
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.15 / 6.0
    
    # AlCamarilla levels to 12h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_1w_aligned[i]
        s1 = camarilla_s1_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r1 = camarilla_r1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price at S1 or S3 + uptrend + volume spike
            if ((abs(price - s1) < 0.001 * price or abs(price - s3) < 0.001 * price) and 
                price > ema_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short conditions: price at R1 or R3 + downtrend + volume spike
            elif ((abs(price - r1) < 0.001 * price or abs(price - r3) < 0.001 * price) and 
                  price < ema_val and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches opposite pivot (R1) or trend breaks
                if price >= r1 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches opposite pivot (S1) or trend breaks
                if price <= s1 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_S1S3_R1R3_1wEMA_Trend_Volume"
timeframe = "12h"
leverage = 1.0