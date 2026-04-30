#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R4 level with 1w uptrend (close > 1w EMA50) and volume > 1.8x 20-bar avg.
# Short when price breaks below Camarilla S4 level with 1w downtrend (close < 1w EMA50) and volume > 1.8x 20-bar avg.
# Exit on opposite Camarilla level touch (mean reversion within the pivot structure).
# Uses 1w EMA50 for stronger trend filtering to reduce trades and avoid SOL-only bias.
# Timeframe: 4h, HTF: 1w as per experiment guidelines.

name = "4h_Camarilla_R4S4_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Previous 1w OHLC for completed 1w bar (no look-ahead)
    df_1w_prev = get_htf_data(prices, '1w')
    if len(df_1w_prev) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w_prev['high'].shift(1).values
    prev_low_1w = df_1w_prev['low'].shift(1).values
    prev_close_1w = df_1w_prev['close'].shift(1).values
    
    # Align 1w data to 4h timeframe (completed 1w bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_high_1w)
    prev_low_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_low_1w)
    prev_close_aligned = align_htf_to_ltf(prices, df_1w_prev, prev_close_1w)
    
    # Camarilla pivot levels from previous completed 1w bar (no look-ahead)
    # R4 = close + 1.1*(high - low), S4 = close - 1.1*(high - low)
    camarilla_r4 = prev_close_aligned + 1.1 * (prev_high_aligned - prev_low_aligned)
    camarilla_s4 = prev_close_aligned - 1.1 * (prev_high_aligned - prev_low_aligned)
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_camarilla_r4 = camarilla_r4[i]
        curr_camarilla_s4 = camarilla_s4[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R4, uptrend (close > 1w EMA50), volume spike
            if (curr_close > curr_camarilla_r4 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4, downtrend (close < 1w EMA50), volume spike
            elif (curr_close < curr_camarilla_s4 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla S4 (mean reversion)
            if curr_close <= curr_camarilla_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla R4 (mean reversion)
            if curr_close >= curr_camarilla_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals