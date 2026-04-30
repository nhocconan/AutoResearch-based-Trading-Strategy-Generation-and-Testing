#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
# Uses proven Camarilla structure from daily pivots (more stable than intraday) with strict volume confirmation.
# 6h timeframe targets 12-37 trades/year to avoid fee drag. Works in bull/bear via 1d EMA34 trend filter.
# Long when price breaks above daily R3 with 1d uptrend and volume > 2.0x 20-bar avg.
# Short when price breaks below daily S3 with 1d downtrend and volume > 2.0x 20-bar avg.
# Exit on opposite Camarilla level (R2 for longs, S2 for shorts) for mean reversion within the pivot structure.

name = "6h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d OHLC for completed 1d bar (no look-ahead) - used for Camarilla calculation
    df_1d_prev = df_1d.copy()
    prev_high_1d = df_1d_prev['high'].shift(1).values
    prev_low_1d = df_1d_prev['low'].shift(1).values
    prev_close_1d = df_1d_prev['close'].shift(1).values
    
    # Align 1d data to 6h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_close_1d)
    
    # Calculate Camarilla pivot levels from previous completed 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6)
    #          S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    H = prev_high_aligned
    L = prev_low_aligned
    C = prev_close_aligned
    diff = H - L
    
    # Avoid division by zero and handle NaN
    camarilla_r3 = C + (diff * 1.1 / 4)
    camarilla_s3 = C - (diff * 1.1 / 4)
    camarilla_r2 = C + (diff * 1.1 / 6)
    camarilla_s2 = C - (diff * 1.1 / 6)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r2[i]) or np.isnan(camarilla_s2[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, uptrend (close > 1d EMA34), volume spike
            if (curr_close > camarilla_r3[i] and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, downtrend (close < 1d EMA34), volume spike
            elif (curr_close < camarilla_s3[i] and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla R2 (mean reversion)
            if curr_close >= camarilla_r2[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla S2 (mean reversion)
            if curr_close <= camarilla_s2[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals