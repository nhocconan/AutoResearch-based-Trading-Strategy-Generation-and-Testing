#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Camarilla pivots provide intraday support/resistance levels based on previous day's range.
# Breakout above R3 or below S3 with 4h trend alignment captures strong moves.
# Volume confirmation filters for institutional participation; discrete sizing (0.20) minimizes fee churn.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session filter (UTC 08-20)
    hours = prices.index.hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla pivots (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R3 = High + 1.1 * (High - Low) / 2
    # S3 = Low - 1.1 * (High - Low) / 2
    camarilla_r3 = prev_high + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_low - 1.1 * (prev_high - prev_low) / 2
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation (on 1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 4h EMA50, volume MA warmup
    
    for i in range(start_idx, n):
        # Session filter: only trade UTC 08-20
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Breakout conditions
        breakout_long = curr_high > curr_r3  # Price breaks above R3
        breakout_short = curr_low < curr_s3  # Price breaks below S3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA50 OR opposite breakout (break below S3)
            if curr_close < curr_ema_4h or breakout_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50 OR opposite breakout (break above R3)
            if curr_close > curr_ema_4h or breakout_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: breakout above R3 AND above 4h EMA50 AND volume confirmation
            if (breakout_long and 
                curr_close > curr_ema_4h and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: breakout below S3 AND below 4h EMA50 AND volume confirmation
            elif (breakout_short and 
                  curr_close < curr_ema_4h and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals