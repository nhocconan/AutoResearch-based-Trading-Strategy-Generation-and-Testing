#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level with 12h uptrend (close > 12h EMA50) and volume > 2.0x 20-bar avg.
# Short when price breaks below Camarilla S3 level with 12h downtrend (close < 12h EMA50) and volume > 2.0x 20-bar avg.
# Exit on opposite Camarilla level touch (mean reversion within the pivot structure).
# Uses proven Camarilla pivot structure with strict volume confirmation (2.0x) and 12h EMA50 trend filter to limit trades.
# 12h EMA50 provides longer-term trend filter, reducing false signals in choppy markets and bear rallies.
# Timeframe: 6h, HTF: 12h as per experiment guidelines.

name = "6h_Camarilla_R3S3_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous 12h OHLC for completed 12h bar (no look-ahead)
    df_12h_prev = get_htf_data(prices, '12h')
    if len(df_12h_prev) < 2:
        return np.zeros(n)
    
    prev_high_12h = df_12h_prev['high'].shift(1).values
    prev_low_12h = df_12h_prev['low'].shift(1).values
    prev_close_12h = df_12h_prev['close'].shift(1).values
    
    # Align 12h data to 6h timeframe (completed 12h bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_12h_prev, prev_high_12h)
    prev_low_aligned = align_htf_to_ltf(prices, df_12h_prev, prev_low_12h)
    prev_close_aligned = align_htf_to_ltf(prices, df_12h_prev, prev_close_12h)
    
    # Camarilla pivot levels from previous completed 12h bar (no look-ahead)
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    camarilla_r3 = prev_close_aligned + 1.1 * (prev_high_aligned - prev_low_aligned) / 2
    camarilla_s3 = prev_close_aligned - 1.1 * (prev_high_aligned - prev_low_aligned) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_camarilla_r3 = camarilla_r3[i]
        curr_camarilla_s3 = camarilla_s3[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, uptrend (close > 12h EMA50), volume spike
            if (curr_close > curr_camarilla_r3 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, downtrend (close < 12h EMA50), volume spike
            elif (curr_close < curr_camarilla_s3 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla S3 (mean reversion)
            if curr_close <= curr_camarilla_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla R3 (mean reversion)
            if curr_close >= curr_camarilla_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals