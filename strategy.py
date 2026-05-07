#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Daily high/low for Camarilla levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3)
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low) / 6
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low) / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with uptrend and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_34_aligned[i] > ema_34_aligned[i-1] and  # Rising EMA
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with downtrend and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_34_aligned[i] < ema_34_aligned[i-1] and  # Falling EMA
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below S3 or trend turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                ema_34_aligned[i] < ema_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above R3 or trend turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                ema_34_aligned[i] > ema_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong support/resistance in 4h timeframe.
# 1d EMA(34) provides trend filter to avoid counter-trend trades.
# Volume confirmation ensures institutional participation.
# Long when price breaks above R3 in uptrend with volume.
# Short when price breaks below S3 in downtrend with volume.
# Exits on trend reversal or price retracement to opposite S3/R3 level.
# Designed for 15-25 trades/year to minimize fee drag while capturing significant moves.