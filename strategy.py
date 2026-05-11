#!/usr/bin/env python3
# 12h_1w_Camarilla_Pivot_R3_S3_Breakout_Trend
# Hypothesis: Combines 1w trend filter with 12h Camarilla R3/S3 breakouts and volume confirmation.
# In bull markets: 1w uptrend + breakout above R3 captures strong momentum.
# In bear markets: 1w downtrend + breakdown below S3 captures accelerated moves.
# Volume surge filters low-momentum breakouts. Target: 15-30 trades/year to minimize fee drag.

name = "12h_1w_Camarilla_Pivot_R3_S3_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get multi-timeframe data
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w Close for trend filter ---
    close_1w = df_1w['close'].values
    
    # --- 12h Camarilla levels (R3, S3) from previous 12h bar ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    prev_12h_high = df_12h['high'].values
    prev_12h_low = df_12h['low'].values
    prev_12h_close = df_12h['close'].values
    
    camarilla_width = (prev_12h_high - prev_12h_low) * 1.1 / 2.0
    camarilla_r3 = prev_12h_close + camarilla_width
    camarilla_s3 = prev_12h_close - camarilla_width
    
    # Align 12h Camarilla levels to 12h timeframe (no alignment needed as we're already at 12h)
    # But we need to align the 1w trend to 12h timeframe
    # Since we're using 12h as primary timeframe, we need to align 1w data to 12h
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Align 12h Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # --- Volume confirmation (2.0x 24-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 24-period volume MA
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(close_1w_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and 1w uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                close_1w_aligned[i] < close[i]):  # 1w uptrend: current price above weekly close
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge and 1w downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  close_1w_aligned[i] > close[i]):  # 1w downtrend: current price below weekly close
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR 1w trend turns down
                if (close[i] < camarilla_s3_aligned[i] or 
                    close_1w_aligned[i] > close[i]):  # 1w turned down
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR 1w trend turns up
                if (close[i] > camarilla_r3_aligned[i] or 
                    close_1w_aligned[i] < close[i]):  # 1w turned up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals