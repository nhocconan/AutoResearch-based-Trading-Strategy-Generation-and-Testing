#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3, 4h EMA50 uptrend, volume > 2.0x 20-bar avg, and UTC hour 8-20.
# Short when price breaks below S3, 4h EMA50 downtrend, volume > 2.0x 20-bar avg, and UTC hour 8-20.
# Exit on touch of S3 (for longs) or R3 (for shorts).
# Uses 4h/1d for signal direction (trend, pivots) and 1h only for entry timing and session filter.
# Discrete position size 0.20 to minimize fee churn. Target 15-37 trades/year.

name = "1h_Camarilla_R3S3_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (UTC 8-20) using DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 4h bar (use previous completed 4h period)
    high_shift = df_4h['high'].shift(1).values
    low_shift = df_4h['low'].shift(1).values
    close_shift = df_4h['close'].shift(1).values
    
    # Align the prior 4h bar's OHLC to 1h timeframe
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_shift)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_shift)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_shift)
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_range = high_4h_aligned - low_4h_aligned
    r3 = close_4h_aligned + camarilla_range * 1.1 / 4
    s3 = close_4h_aligned - camarilla_range * 1.1 / 4
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r3 = r3[i]
        curr_s3 = s3[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, uptrend (close > 4h EMA50), volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, downtrend (close < 4h EMA50), volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches or goes below S3 (mean reversion)
            if curr_close <= curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price touches or goes above R3 (mean reversion)
            if curr_close >= curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals