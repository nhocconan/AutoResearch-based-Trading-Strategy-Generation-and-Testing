#!/usr/bin/env python3
# 4h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Uses 1-day Camarilla R3/S3 levels with trend filter from 1d EMA34 and volume confirmation.
# In bull markets: price breaks above R3 with volume and uptrend triggers long.
# In bear markets: price breaks below S3 with volume and downtrend triggers short.
# Camarilla levels provide strong intraday support/resistance that work in ranging and trending markets.
# Target: 20-50 trades/year to minimize fee drag while capturing meaningful moves.

name = "4h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day Camarilla levels (R3, S3) from previous day ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = close + 1.1*(high-low)/6, S3 = close - 1.1*(high-low)/6
    r3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    s3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Shift by 1 to use only completed 1-day candle (avoid look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan  # First value invalid after roll
    s3_shifted[0] = np.nan
    
    # Align 1-day Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    
    # --- 1-day EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume confirmation (1.5x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1-day EMA34 (34 periods) and 20-period volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume surge and 1-day uptrend
            if (close[i] > r3_aligned[i] and 
                volume_surge and 
                ema_34_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume surge and 1-day downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_surge and 
                  ema_34_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S3 OR 1-day EMA34 turns down
                if (close[i] < s3_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R3 OR 1-day EMA34 turns up
                if (close[i] > r3_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals