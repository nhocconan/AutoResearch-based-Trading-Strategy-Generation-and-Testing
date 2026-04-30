#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Long when price breaks above R3 with 1w uptrend (price > 1w EMA50) and volume spike.
# Short when price breaks below S3 with 1w downtrend (price < 1w EMA50) and volume spike.
# Exit when price returns to the Camarilla H3/L3 level (mean reversion to midpoint).
# Camarilla levels provide institutional support/resistance; 1w EMA50 filters trend; volume confirms participation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeConfirmation_v1"
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
    
    # Calculate Camarilla levels from previous 1d bar
    # Need 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d close, high, low (already completed bar)
    prev_close = df_1d['close'].shift(1).values  # shift(1) for previous completed bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align 1d data to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # H3 = close + (high-low)*1.1/6
    # L3 = close - (high-low)*1.1/6
    range_1d = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_aligned - range_1d * 1.1 / 4
    camarilla_h3 = prev_close_aligned + range_1d * 1.1 / 6
    camarilla_l3 = prev_close_aligned - range_1d * 1.1 / 6
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_h3 = camarilla_h3[i]
        curr_l3 = camarilla_l3[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, uptrend (price > 1w EMA50), volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, downtrend (price < 1w EMA50), volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to H3 (mean reversion)
            if curr_close <= curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to L3 (mean reversion)
            if curr_close >= curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals