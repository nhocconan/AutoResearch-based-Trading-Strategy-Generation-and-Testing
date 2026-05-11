#!/usr/bin/env python3
# 1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Uses 1-week Camarilla R3/S3 levels for breakout with 1-day trend filter and volume confirmation on daily timeframe.
# In bull markets: 1w uptrend + breakout above R3 captures momentum.
# In bear markets: 1w downtrend + breakdown below S3 captures short opportunities.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Target: 10-30 trades/year to minimize fee drag while capturing meaningful moves.

name = "1d_1w_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-week data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-week Camarilla levels (R3, S3) from previous candle ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Shift by 1 to use only completed 1w candle (avoid look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan  # First value invalid after roll
    camarilla_s3[0] = np.nan
    
    # Align 1w Camarilla levels to 1d
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # --- 1-week EMA50 for trend filter ---
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Volume confirmation (1.5x 20-period average on 1d) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1w EMA50 (50 periods) and 20-period volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume surge and 1w uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                ema_50_1w_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume surge and 1w downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  ema_50_1w_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below Camarilla S3 OR 1w EMA50 turns down
                if (close[i] < camarilla_s3_aligned[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above Camarilla R3 OR 1w EMA50 turns up
                if (close[i] > camarilla_r3_aligned[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals