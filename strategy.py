#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (S3/R3) breakout with volume confirmation (>1.5x 20-period EMA) and 1-day EMA34 trend filter.
# Camarilla levels provide precise intraday support/resistance; volume confirms breakout conviction; 1d EMA34 ensures alignment with daily trend.
# Works in bull markets (breaks above R3) and bear markets (breaks below S3).
# Uses proven Camarilla pivot structure from top-performing strategies.
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4
    # R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    camarilla_S3 = typical_price - (range_hl * 1.1 / 4)
    camarilla_R3 = typical_price + (range_hl * 1.1 / 4)
    
    # 1-day EMA34 trend filter
    ema_1d = df_1d['close'].ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3.values)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3.values)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: close > Camarilla R3 + volume confirmation + 1d EMA34 up
            if (price > camarilla_R3_aligned[i] and vol_confirm[i] and price > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close < Camarilla S3 + volume confirmation + 1d EMA34 down
            elif (price < camarilla_S3_aligned[i] and vol_confirm[i] and price < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close crosses below Camarilla S3
            if price < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close crosses above Camarilla R3
            if price > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals