#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 level with 4h uptrend (close > 4h EMA50) and volume > 1.8x 24-bar avg.
# Short when price breaks below Camarilla S1 level with 4h downtrend (close < 4h EMA50) and volume > 1.8x 24-bar avg.
# Exit on opposite Camarilla level touch (mean reversion within the pivot structure).
# Uses 4h EMA50 for trend filter to reduce false signals, volume confirmation for conviction, and 1h timeframe for precise entry timing.
# Session filter (08-20 UTC) applied to reduce noise trades. Target signal size: 0.20.

name = "1h_Camarilla_R1S1_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 4h OHLC for completed 4h bar (no look-ahead)
    df_4h_prev = get_htf_data(prices, '4h')
    if len(df_4h_prev) < 2:
        return np.zeros(n)
    
    prev_high_4h = df_4h_prev['high'].shift(1).values
    prev_low_4h = df_4h_prev['low'].shift(1).values
    prev_close_4h = df_4h_prev['close'].shift(1).values
    
    # Align 4h data to 1h timeframe (completed 4h bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_4h_prev, prev_high_4h)
    prev_low_aligned = align_htf_to_ltf(prices, df_4h_prev, prev_low_4h)
    prev_close_aligned = align_htf_to_ltf(prices, df_4h_prev, prev_close_4h)
    
    # Camarilla pivot levels from previous completed 4h bar (no look-ahead)
    # R1 = close + 1.1*(high - low)/12, S1 = close - 1.1*(high - low)/12
    camarilla_r1 = prev_close_aligned + 1.1 * (prev_high_aligned - prev_low_aligned) / 12
    camarilla_s1 = prev_close_aligned - 1.1 * (prev_high_aligned - prev_low_aligned) / 12
    
    # Volume confirmation: volume > 1.8x 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(volume_confirm[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_camarilla_r1 = camarilla_r1[i]
        curr_camarilla_s1 = camarilla_s1[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R1, uptrend (close > 4h EMA50), volume spike
            if (curr_close > curr_camarilla_r1 and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1, downtrend (close < 4h EMA50), volume spike
            elif (curr_close < curr_camarilla_s1 and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla S1 (mean reversion)
            if curr_close <= curr_camarilla_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla R1 (mean reversion)
            if curr_close >= curr_camarilla_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals