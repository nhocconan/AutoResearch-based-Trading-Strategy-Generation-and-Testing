#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND close < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses 1w EMA34 (trend change)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Camarilla R3/S3 levels provide strong support/resistance for filtered entries.
# Weekly EMA34 aligns with major trend, volume confirmation ensures participation.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.

name = "6h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivots (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d bar
    # Camarilla formula: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 6h timeframe (use previous 1d bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 50)  # volume MA, EMA34, and data warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 1w EMA34 (trend change)
            if curr_close < curr_ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1w EMA34 (trend change)
            if curr_close > curr_ema34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1w EMA34 AND volume confirmation
            if curr_close > curr_r3 and curr_close > curr_ema34_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND close < 1w EMA34 AND volume confirmation
            elif curr_close < curr_s3 and curr_close < curr_ema34_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals