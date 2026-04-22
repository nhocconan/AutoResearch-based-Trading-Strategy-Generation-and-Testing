#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot reversal with 1-day EMA trend filter and volume confirmation.
Trades reversals at Camarilla R3/S3 levels when price shows rejection (close outside level but open inside).
Uses 1-day EMA34 for trend filter and volume spike for confirmation. Designed for low trade frequency
(15-30 trades/year) to minimize fee drag and work in both bull and bear markets by fading extremes
in the direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3, S3 (most significant for reversals)
    # R3 = close + 1.1*(high-low)*1.1/2
    # S3 = close - 1.1*(high-low)*1.1/2
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev) * 1.1 / 2
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev) * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long reversal at S3: open below S3, close above S3 (bullish rejection)
            if open_price[i] < camarilla_s3_aligned[i] and close[i] > camarilla_s3_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal at R3: open above R3, close below R3 (bearish rejection)
            elif open_price[i] > camarilla_r3_aligned[i] and close[i] < camarilla_r3_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R3 or closes below 1d EMA
                if close[i] >= camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S3 or closes above 1d EMA
                if close[i] <= camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Reversal_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0