#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA20 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability daily support/resistance
# 1w EMA20 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation requires 2.0x average volume to ensure participation while avoiding overtrading
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by following the 1w trend direction

name = "1d_Camarilla_R3S3_1wEMA20_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # We need previous day's OHLC, so we shift by 1
    prev_close = close[1:] if len(close) > 1 else np.array([np.nan])
    prev_high = high[1:] if len(high) > 1 else np.array([np.nan])
    prev_low = low[1:] if len(low) > 1 else np.array([np.nan])
    
    # Pad with NaN for first bar
    prev_close = np.concatenate([[np.nan], prev_close])
    prev_high = np.concatenate([[np.nan], prev_high])
    prev_low = np.concatenate([[np.nan], prev_low])
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (strict to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout signals with 1w trend filter
        # Long: price breaks above Camarilla R3 + volume spike + price above 1w EMA20
        # Short: price breaks below Camarilla S3 + volume spike + price below 1w EMA20
        if position == 0:
            if (close[i] > camarilla_r3[i] and volume_spike and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_s3[i] and volume_spike and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (reversal) OR price below 1w EMA20
            if close[i] < camarilla_s3[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (reversal) OR price above 1w EMA20
            if close[i] > camarilla_r3[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals