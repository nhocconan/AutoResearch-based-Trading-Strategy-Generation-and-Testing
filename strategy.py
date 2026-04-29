#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 2.0x 24-bar avg
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 2.0x 24-bar avg
# Exit when price crosses 4h EMA50 (trend change)
# Uses discrete position sizing (0.20) to control risk and minimize fee churn.
# Session filter: only trade 08-20 UTC to avoid low-liquidity periods.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Camarilla R3/S3 provides tighter breakout levels than R1/S1 for precise entries.
# 4h EMA50 aligns with medium-term trend, volume confirmation ensures participation.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivots (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d bar
    # Camarilla formula: R3 = close + 1.1*(high-low)/6, S3 = close - 1.1*(high-low)/6
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 6
    s3 = close_1d - 1.1 * camarilla_range / 6
    
    # Align Camarilla levels to 1h timeframe (use previous 1d bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: >2.0x 24-bar average volume (stricter to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA50 (trend change)
            if curr_close < curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA50 (trend change)
            if curr_close > curr_ema50_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 4h EMA50 AND volume confirmation
            if curr_close > curr_r3 and curr_close > curr_ema50_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 AND close < 4h EMA50 AND volume confirmation
            elif curr_close < curr_s3 and curr_close < curr_ema50_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals