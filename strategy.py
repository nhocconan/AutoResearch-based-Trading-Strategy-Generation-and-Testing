#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter (HMA21) and volume confirmation.
# Long when price breaks above Camarilla R3 in 1w uptrend (price > HMA21).
# Short when price breaks below Camarilla S3 in 1w downtrend (price < HMA21).
# Volume must be > 1.5x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 30-100 total trades over 4 years.

name = "1d_Camarilla_R3S3_1wHMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w HMA21
    close_1w = df_1w['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = pd.Series(close_1w).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    hma_21_1w = wma_sqrt
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Camarilla levels on 1d (based on previous day's range)
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # start after warmup for Camarilla calculation
        # Skip if any value is NaN
        if (np.isnan(hma_21_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > hma_21_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < hma_21_1w_aligned[i]  # 1w downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1w uptrend AND volume spike
            if close_val > camarilla_r3[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND 1w downtrend AND volume spike
            elif close_val < camarilla_s3[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR 1w trend turns down
            if close_val < camarilla_s3[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR 1w trend turns up
            if close_val > camarilla_r3[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals