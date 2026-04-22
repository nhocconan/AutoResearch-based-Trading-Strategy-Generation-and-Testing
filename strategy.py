#!/usr/bin/env python3

"""
Hypothesis: 12-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation.
Trades reversals at Camarilla S3/R3 levels in counter-trend direction when 1d EMA trend is weak.
Uses volume spike confirmation to avoid false signals. Designed for low trade frequency
(12-37 trades/year) to minimize fee drift and work in both bull and bear markets by
fading extremes in ranging conditions while respecting stronger daily trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
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
    
    # Calculate Camarilla levels from previous 1d bar
    # Using typical formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use R3 and S3 as our key levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_camarilla = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = close_1d_for_camarilla + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d_for_camarilla - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price at or below S3 with weak downtrend (price above 1d EMA)
            if low[i] <= camarilla_s3_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or above R3 with weak uptrend (price below 1d EMA)
            elif high[i] >= camarilla_r3_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend strengthens
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R3 or closes below 1d EMA (trend turning down)
                if high[i] >= camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S3 or closes above 1d EMA (trend turning up)
                if low[i] <= camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_S3R3_Reversal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0