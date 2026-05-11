#!/usr/bin/env python3
# 1h_4d_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Uses 4-day trend filter (1-day EMA vs 4-day EMA) with 1-hour Camarilla R1/S1 breakouts and volume confirmation.
# In bull markets: 4-day uptrend + 1h breakout above R1 with volume surge = long.
# In bear markets: 4-day downtrend + 1h breakdown below S1 with volume surge = short.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Target: 15-35 trades/year to minimize fee drag while capturing meaningful moves.
# Uses 4h for trend alignment and 1d for Camarilla calculation to avoid overtrading.

name = "1h_4d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4-day trend: EMA50 on 4h (approx 4 days of 4h candles) ---
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- Daily Camarilla levels (R1, S1) from previous day ---
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    
    camarilla_width = (prev_1d_high - prev_1d_low) * 1.1 / 6.0  # R1/S1 level
    camarilla_r1 = prev_1d_close + camarilla_width
    camarilla_s1 = prev_1d_close - camarilla_width
    
    # Align daily Camarilla levels to 1h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # --- Volume confirmation (1.5x 48-period average on 1h) ---
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 4h EMA50 (50 periods) and 48-period volume MA
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
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and 4-day uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_surge and 
                ema_50_4h_aligned[i] < close[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume surge and 4-day downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_surge and 
                  ema_50_4h_aligned[i] > close[i]):
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1 OR 4-day EMA50 turns down
                if (close[i] < camarilla_s1_aligned[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price rises above R1 OR 4-day EMA50 turns up
                if (close[i] > camarilla_r1_aligned[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals