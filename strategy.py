#!/usr/bin/env python3
# 4h_12h_1d_Camarilla_R3_S3_Breakout_Trend_MultiTF
# Hypothesis: Combining 12h and 1d timeframe confluence with Camarilla R3/S3 breakouts
# creates higher-probability trades. In bull markets, 1d uptrend + 12h breakout above R3
# signals strong momentum. In bear markets, 1d downtrend + 12h breakdown below S3
# captures accelerated moves. Volume confirmation filters low-momentum breakouts.
# Multi-timeframe alignment reduces false signals and improves trade quality.
# Target: 20-40 trades/year to minimize fee drag while capturing meaningful moves.

name = "4h_12h_1d_Camarilla_R3_S3_Breakout_Trend_MultiTF"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get multi-timeframe data
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Camarilla levels (R3, S3) from previous 12h bar ---
    prev_12h_high = df_12h['high'].values
    prev_12h_low = df_12h['low'].values
    prev_12h_close = df_12h['close'].values
    
    camarilla_width = (prev_12h_high - prev_12h_low) * 1.1 / 2.0
    camarilla_r3 = prev_12h_close + camarilla_width
    camarilla_s3 = prev_12h_close - camarilla_width
    
    # Align 12h Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # --- 12h EMA50 for trend filter ---
    ema_50_12h = pd.Series(prev_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- 1d EMA200 for higher timeframe trend filter ---
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- Volume confirmation (2.5x 30-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d EMA200 (200 periods) and 30-period volume MA
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge, 12h EMA50 uptrend, and 1d EMA200 uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                ema_50_12h_aligned[i] < close[i] and 
                ema_200_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge, 12h EMA50 downtrend, and 1d EMA200 downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  ema_50_12h_aligned[i] > close[i] and 
                  ema_200_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR 12h EMA50 turns down OR 1d EMA200 turns down
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_50_12h_aligned[i] or 
                    close[i] < ema_200_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR 12h EMA50 turns up OR 1d EMA200 turns up
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_50_12h_aligned[i] or 
                    close[i] > ema_200_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals