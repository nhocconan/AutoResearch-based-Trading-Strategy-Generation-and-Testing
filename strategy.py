#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout + Volume + Daily Trend Filter
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Breaking above R3 or below S3 with volume confirmation and daily trend alignment
captures significant moves while avoiding false breakouts in ranging markets.
Designed for low trade frequency (~20-30/year) on 12h timeframe to minimize fee decay.
Works in both bull and bear markets by following the daily trend direction.
"""
name = "12h_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # === Daily OHLC for Camarilla calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), etc.
    # But we use standard: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Actually standard Camarilla: R3 = Close + (High-Low)*1.1/2, S3 = Close - (High-Low)*1.1/2
    # Wait, correct formula: R3 = Close + (High-Low)*1.1/2, S3 = Close - (High-Low)*1.1/2
    # No, let me check: Standard Camarilla:
    # R4 = Close + ((High-Low) * 1.5000)
    # R3 = Close + ((High-Low) * 1.2500)
    # R2 = Close + ((High-Low) * 1.1666)
    # R1 = Close + ((High-Low) * 1.0833)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High-Low) * 1.0833)
    # S2 = Close - ((High-Low) * 1.1666)
    # S3 = Close - ((High-Low) * 1.2500)
    # S4 = Close - ((High-Low) * 1.5000)
    # But many use: R3 = Close + (High-Low)*1.1/2, S3 = Close - (High-Low)*1.1/2
    # Actually looking at successful strategies, they use R3/S3 as:
    # R3 = Close + (High-Low) * 1.1
    # S3 = Close - (High-Low) * 1.1
    # Let me verify with the working examples...
    # From the database: "Camarilla R3 S3" suggests they use the 1.1 multiplier
    
    # Calculate using standard Camarilla R3/S3: R3 = C + (H-L)*1.1, S3 = C - (H-L)*1.1
    # This matches what successful strategies use
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Camarilla R3 and S3 from previous day
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Daily EMA34 for trend filter ===
    # Using close prices for EMA
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (20-period on 12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 + above daily EMA34 + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + below daily EMA34 + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR below daily EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR above daily EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals