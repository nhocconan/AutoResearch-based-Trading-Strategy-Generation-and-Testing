#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Camarilla R3/S3 levels act as strong support/resistance from 1d timeframe.
# Breakout above R3 or below S3 with volume confirmation and 12h EMA50 trend alignment
# captures institutional breakout attempts. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h and 1d calculations
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d OHLC for Camarilla pivot levels (R3, S3, R4, S4)
    # Camarilla formula: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #                  S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    cam_r3_1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    cam_s3_1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    cam_r4_1d = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    cam_s4_1d = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 6h timeframe (use completed 1d bar)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3_1d.values)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3_1d.values)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4_1d.values)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4_1d.values)
    
    # Volume confirmation: volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(cam_r3_aligned[i]) or 
            np.isnan(cam_s3_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_cam_r3 = cam_r3_aligned[i]
        curr_cam_s3 = cam_s3_aligned[i]
        curr_cam_r4 = cam_r4_aligned[i]
        curr_cam_s4 = cam_s4_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price crosses below 12h EMA50 (trend change)
            # 2. Price re-enters Camarilla R3-S3 range (breakout failed)
            if (curr_close < curr_ema_50_12h or
                curr_close < curr_cam_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price crosses above 12h EMA50 (trend change)
            # 2. Price re-enters Camarilla R3-S3 range (breakout failed)
            if (curr_close > curr_ema_50_12h or
                curr_close > curr_cam_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + above 12h EMA50 + volume confirm
            if (curr_close > curr_cam_r3 and
                curr_close > curr_ema_50_12h and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Camarilla S3 + below 12h EMA50 + volume confirm
            elif (curr_close < curr_cam_s3 and
                  curr_close < curr_ema_50_12h and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals