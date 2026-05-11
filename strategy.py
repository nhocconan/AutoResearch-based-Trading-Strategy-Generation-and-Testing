#!/usr/bin/env python3
# 1h_4d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Uses 1-day trend filter with 4-hour Camarilla R3/S3 breakouts and volume confirmation.
# 1-hour timeframe for precise entry timing, with 4-hour HTF for signal direction.
# In bull markets: daily uptrend + 4h breakout above R3 + volume surge = long.
# In bear markets: daily downtrend + 4h breakdown below S3 + volume surge = short.
# Volume filter reduces false signals. Target: 15-37 trades/year to avoid fee drag.

name = "1h_4d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4-hour data for trend filter and Camarilla calculation (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA50 for trend filter ---
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 4-hour Camarilla levels (R3, S3) from previous 4h bar ---
    prev_4h_high = df_4h['high'].values
    prev_4h_low = df_4h['low'].values
    prev_4h_close = df_4h['close'].values
    
    camarilla_width = (prev_4h_high - prev_4h_low) * 1.1 / 2.0
    camarilla_r3 = prev_4h_close + camarilla_width
    camarilla_s3 = prev_4h_close - camarilla_width
    
    # Align 4h Camarilla levels to 1h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # --- Volume confirmation (2x 24-period average on 1h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # --- Session filter: 08-20 UTC ---
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 4h EMA50 (50 periods) and 24-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and 4h uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                ema_50_4h_aligned[i] < close[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume surge and 4h downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  ema_50_4h_aligned[i] > close[i]):
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR 4h EMA50 turns down
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price rises above R3 OR 4h EMA50 turns up
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals