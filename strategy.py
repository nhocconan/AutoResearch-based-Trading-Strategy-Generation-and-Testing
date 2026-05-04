#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 20 EMA volume)
# Uses 12h Camarilla pivot levels (R3/S3) for structure - captures strong momentum at key support/resistance
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>1.8x average volume) - tighter to reduce trades to 12-37/year target
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets (continuation at R3) and bear markets (continuation at S3)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R3S3_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h Camarilla levels (R3, S3) from prior completed 12h bar
    # Typical price for Camarilla calculation
    typical_price = (high + low + close) / 3.0
    
    # Calculate pivot and ranges using rolling window of 1 period (previous bar)
    # For 12h timeframe, we use the previous completed 12h bar's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Range = H - L
    range_hl = prev_high - prev_low
    
    # Camarilla levels:
    # R3 = PP + (Range * 1.1/4) = PP + Range * 0.275
    # S3 = PP - (Range * 1.1/4) = PP - Range * 0.275
    r3 = pp + (range_hl * 0.275)
    s3 = pp - (range_hl * 0.275)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1d EMA50 AND volume spike
            if close[i] > r3[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND price < 1d EMA50 AND volume spike
            elif close[i] < s3[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1d EMA50
            if close[i] < s3[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1d EMA50
            if close[i] > r3[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals