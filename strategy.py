#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Targets 6h timeframe with ~12-37 trades/year. Long when price breaks above Camarilla R3 with volume and price > 12h EMA50.
# Short when price breaks below Camarilla S3 with volume and price < 12h EMA50.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear via trend filter.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter and Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    # We use the previous completed 12h bar to avoid look-ahead
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(len(df_12h)):
        H = high_12h[i]
        L = low_12h[i]
        C = close_12h[i]
        camarilla_r3_val = C + ((H - L) * 1.1 / 4)
        camarilla_s3_val = C - ((H - L) * 1.1 / 4)
        # Align to LTF: this value becomes available after the 12h bar closes
        start_idx = i * 2  # 2x 6h bars per 12h bar
        end_idx = min((i + 1) * 2, n)
        camarilla_r3[start_idx:end_idx] = camarilla_r3_val
        camarilla_s3[start_idx:end_idx] = camarilla_s3_val
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Camarilla R3/S3 breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r3[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s3[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level (S3 for long, R3 for short) or trend reversal
        long_exit = close[i] < camarilla_s3[i] or close[i] < ema_50_12h_aligned[i]
        short_exit = close[i] > camarilla_r3[i] or close[i] > ema_50_12h_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals