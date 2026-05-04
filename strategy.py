#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses 1d Camarilla pivot levels (R3/S3) for structure - captures strong momentum bursts
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.5x average volume) - balanced to reduce trades
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets (continuation at R3) and bear markets (continuation at S3)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "1d_Camarilla_R3S3_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Camarilla levels from prior completed 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using typical formula: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # But standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    high_low = high - low
    camarilla_r3 = close + (high_low * 1.1 / 4)
    camarilla_s3 = close - (high_low * 1.1 / 4)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(camarilla_r3_shifted[i]) or np.isnan(camarilla_s3_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1w EMA50 AND volume spike
            if close[i] > camarilla_r3_shifted[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1w EMA50 AND volume spike
            elif close[i] < camarilla_s3_shifted[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1w EMA50
            if close[i] < camarilla_s3_shifted[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1w EMA50
            if close[i] > camarilla_r3_shifted[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals