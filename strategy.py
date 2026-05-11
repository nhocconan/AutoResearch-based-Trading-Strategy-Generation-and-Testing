#!/usr/bin/env python3
# 1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: 1h price breaks above 1d Camarilla R1 or below S1 with volume surge and aligned 4h trend direction.
# Uses 1d for structural levels (R1/S1), 4h for trend filter (EMA50), and 1h for entry timing with volume confirmation.
# Designed for low trade frequency (15-37/year) to minimize fee drag. Works in bull markets (breakouts continue) 
# and bear markets (breakdowns accelerate when trend aligns). Filters out low-momentum breakouts via volume surge.

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and 4h data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    if len(df_1d) < 2 or len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (R1, S1) from previous 1d bar ---
    # Classic Camarilla formula:
    # H, L, C = high, low, close of previous period
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    camarilla_width = (prev_1d_high - prev_1d_low) * 1.1 / 12.0
    camarilla_r1 = prev_1d_close + camarilla_width
    camarilla_s1 = prev_1d_close - camarilla_width
    
    # Align 1d Camarilla levels to 1h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # --- 4h EMA50 for trend filter ---
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- Volume confirmation (2.0x 20-period average on 1h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 4h EMA50 and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and 4h EMA50 uptrend
            if close[i] > camarilla_r1_aligned[i] and volume_surge and ema_50_4h_aligned[i] < close[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume surge and 4h EMA50 downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume_surge and ema_50_4h_aligned[i] > close[i]:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1 OR price crosses below 4h EMA50
                if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price rises above R1 OR price crosses above 4h EMA50
                if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals