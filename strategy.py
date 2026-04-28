#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below S3 AND close < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit on opposite Camarilla level touch (S3 for longs, R3 for shorts) or trend reversal
# Target: 12-37 trades/year via tight entry conditions and trend filter reducing whipsaw
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend confirmation

name = "12h_Camarilla_R3S3_1wEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    shift = 1  # Use previous day's levels
    high_1d_shifted = np.concatenate([np.full(shift, np.nan), high_1d[:-shift]])
    low_1d_shifted = np.concatenate([np.full(shift, np.nan), low_1d[:-shift]])
    close_1d_shifted = np.concatenate([np.full(shift, np.nan), close_1d[:-shift]])
    
    camarilla_range = high_1d_shifted - low_1d_shifted
    r3 = close_1d_shifted + 1.1 * camarilla_range
    s3 = close_1d_shifted - 1.1 * camarilla_range
    
    # Align 1d Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND above 1w EMA50 AND volume confirmation
            if price > r3_val and price > ema_50_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND below 1w EMA50 AND volume confirmation
            elif price < s3_val and price < ema_50_val and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price touches S3 or trend reverses
            if price < s3_val or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price touches R3 or trend reverses
            if price > r3_val or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals