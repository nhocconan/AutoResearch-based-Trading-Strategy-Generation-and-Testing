#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal with 1d Volume Confirmation
# - Uses 1d Camarilla pivot levels (R3, S3) as reversal zones
# - Long when price crosses above S3 with volume spike, short when crosses below R3
# - Volume confirmation: current 1d volume > 1.5x 20-period average
# - Only trade in alignment with 1w trend: price above/below 200-period EMA
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot point = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1
    # S3 = Pivot - (H - L) * 1.1
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r3 = pivot + (prev_high - prev_low) * 1.1
    s3 = pivot - (prev_high - prev_low) * 1.1
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma * 1.5)
    
    # 1w trend filter: 200-period EMA on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Align 1d indicators to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 4h close for crossover detection
    close_4h = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(volume_spike_4h[i]) or np.isnan(ema_200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is in alignment with 1w trend
        bullish_trend = close_4h[i] > ema_200_1w_aligned[i]
        bearish_trend = close_4h[i] < ema_200_1w_aligned[i]
        
        # Volume spike condition
        vol_spike = volume_spike_4h[i] > 0.5
        
        if position == 0:
            # Long entry: price crosses above S3 + volume spike + bullish trend
            if close_4h[i] > s3_4h[i] and close_4h[i-1] <= s3_4h[i-1] and vol_spike and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below R3 + volume spike + bearish trend
            elif close_4h[i] < r3_4h[i] and close_4h[i-1] >= r3_4h[i-1] and vol_spike and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S3 or trend turns bearish
            if close_4h[i] < s3_4h[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R3 or trend turns bullish
            if close_4h[i] > r3_4h[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0