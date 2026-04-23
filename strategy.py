#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above Camarilla R3 (1d) AND 1d close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 (1d) AND 1d close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when price returns to Camarilla R3/S3 level or opposite pivot (R4/S4 for continuation).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Camarilla pivots from 1d provide precise intraday levels; 1d EMA50 ensures alignment with higher-timeframe trend.
Volume confirmation filters weak breakouts. Designed to work in both bull and bear markets by requiring
trend alignment and avoiding counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Using previous day's OHLC to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    cam_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    cam_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 4
    cam_r4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    cam_s4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) * 1.1 / 2
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for EMA50 and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(cam_r3_aligned[i]) or 
            np.isnan(cam_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above R3 with 1d uptrend and volume spike
            if (price > cam_r3_aligned[i] and 
                close_1d[-1] > ema50_1d[-1] if len(close_1d) > 0 else False and  # 1d close > EMA50 (use last known)
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with 1d downtrend and volume spike
            elif (price < cam_s3_aligned[i] and 
                  close_1d[-1] < ema50_1d[-1] if len(close_1d) > 0 else False and  # 1d close < EMA50
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Return to R3/S3 level
            if position == 1 and price <= cam_r3_aligned[i]:
                exit_signal = True
            elif position == -1 and price >= cam_s3_aligned[i]:
                exit_signal = True
            
            # Alternative exit: Break opposite level (continuation)
            elif position == 1 and price >= cam_r4_aligned[i]:
                exit_signal = True  # Take profit at R4
            elif position == -1 and price <= cam_s4_aligned[i]:
                exit_signal = True  # Take profit at S4
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0