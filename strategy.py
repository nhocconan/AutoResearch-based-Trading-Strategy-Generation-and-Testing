#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1, close > 4h EMA50, and volume > 1.8x 24-bar avg.
# Short when price breaks below S1, close < 4h EMA50, and volume > 1.8x 24-bar avg.
# Exit when price re-enters the Camarilla range (between S1 and R1).
# Uses 1h timeframe with 4h/1d for signal direction to balance trade frequency (target: 15-37/year) and timing precision.
# Camarilla levels from 1d OHLC provide institutional support/resistance.
# 4h EMA50 filters for intermediate-term trend alignment.
# Volume confirmation reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_open_1d = df_1d['open'].shift(1).values
    
    typical_price = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_hl = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    R1 = close_1d + range_hl * 1.1 / 12
    S1 = close_1d - range_hl * 1.1 / 12
    R2 = close_1d + range_hl * 1.1 / 6
    S2 = close_1d - range_hl * 1.1 / 6
    R3 = close_1d + range_hl * 1.1 / 4
    S3 = close_1d - range_hl * 1.1 / 4
    R4 = close_1d + range_hl * 1.1 / 2
    S4 = close_1d - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: volume > 1.8x 24-period average (balanced threshold)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_R1 = R1_aligned[i]
        curr_S1 = S1_aligned[i]
        curr_R2 = R2_aligned[i]
        curr_S2 = S2_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_R4 = R4_aligned[i]
        curr_S4 = S4_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1, close > 4h EMA50, volume spike
            if (curr_close > curr_R1 and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, close < 4h EMA50, volume spike
            elif (curr_close < curr_S1 and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters the Camarilla range (below R1)
            if curr_close < curr_R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters the Camarilla range (above S1)
            if curr_close > curr_S1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals