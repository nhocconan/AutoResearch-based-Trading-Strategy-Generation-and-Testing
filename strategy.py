#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation (>1.5x 20 EMA volume)
# Uses Camarilla pivot levels from prior 4h bar for structure - captures mean reversion at extreme levels
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.5x average volume) - balanced to avoid overtrading
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Works in bull markets (breakouts at R3/S3 with trend) and bear markets (reversions at R3/S3 against trend)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "4h_Camarilla_R3S3_12hEMA50_VolumeConfirm_Balanced"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior completed 4h bar
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    camarilla_range = (high_series - low_series) * 1.1
    r3 = close_series + camarilla_range / 4
    s3 = close_series - camarilla_range / 4
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    r3_shifted = np.roll(r3.values, 1)
    s3_shifted = np.roll(s3.values, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3_shifted[i]) or np.isnan(s3_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 12h EMA50 AND volume spike
            if close[i] > r3_shifted[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 12h EMA50 AND volume spike
            elif close[i] < s3_shifted[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 12h EMA50
            if close[i] < s3_shifted[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 12h EMA50
            if close[i] > r3_shifted[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals