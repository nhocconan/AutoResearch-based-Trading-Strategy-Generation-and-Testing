#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 level AND price > 1w EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 level AND price < 1w EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# 1w EMA50 provides strong trend filter for 12h timeframe, avoiding counter-trend entries in choppy markets.
# Volume spike confirms institutional participation in breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for Camarilla levels (based on previous completed day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Shift by 1 to use previous completed day (avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = prev_high - prev_low
    camarilla_pp = prev_close  # Pivot Point
    camarilla_r3 = camarilla_pp + (range_1d * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (range_1d * 1.1 / 4)
    
    # Align to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(51, 20)  # warmup for EMA (50) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3 level, uptrend (price > 1w EMA50), volume confirmation
            if (curr_high > camarilla_r3_aligned[i] and 
                curr_close > ema_50_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 level, downtrend (price < 1w EMA50), volume confirmation
            elif (curr_low < camarilla_s3_aligned[i] and 
                  curr_close < ema_50_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below pivot point (PP)
            if curr_close < camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above pivot point (PP)
            if curr_close > camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals