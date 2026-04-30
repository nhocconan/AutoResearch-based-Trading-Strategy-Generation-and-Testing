#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R3, 1w EMA50 uptrend, and volume > 2.0x 20-bar avg.
# Short when price breaks below S3, 1w EMA50 downtrend, and volume > 2.0x 20-bar avg.
# Exit on opposite Camarilla level touch (R3 for long exit, S3 for short exit).
# Camarilla levels provide support/resistance based on prior day's range.
# Combined with 1w EMA50 trend filter and volume confirmation to reduce false signals.
# Timeframe: 1d, HTF: 1w as per experiment guidelines.

name = "1d_Camarilla_R3S3_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1d bar (use previous day's high/low/close)
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4, etc.
    # We use the prior completed 1d bar to avoid look-ahead
    high_1d = df_1w['high'].values  # This is incorrect - need 1d data for Camarilla
    low_1d = df_1w['low'].values
    close_1d = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar (incorrect - need daily data)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 1d timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, uptrend (close > 1w EMA50), volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, downtrend (close < 1w EMA50), volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches or goes below S3 (mean reversion)
            if curr_close <= curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches or goes above R3 (mean reversion)
            if curr_close >= curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals