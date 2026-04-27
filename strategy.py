#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike
# Works in bull/bear: breakouts capture momentum, 1d trend filters counter-trend fakes,
# volume spike confirms institutional interest. Low-frequency (~20-40 trades/year)
# avoids fee drag. Camarilla levels provide institutional support/resistance.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (using completed daily bar)
    # Camarilla: H/L/C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid look-ahead: use only previous completed day's data
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    rng = prev_high - prev_low
    r3 = prev_close + rng * 1.1 / 4
    s3 = prev_close - rng * 1.1 / 4
    r4 = prev_close + rng * 1.1 / 2
    s4 = prev_close - rng * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for daily bar to complete)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 for trend filter (using completed daily bar)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detector (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup periods
    start_idx = max(34, 20)  # EMA34 needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long breakout: price crosses above R3 with volume, in uptrend
            if (price > r3_6h[i] and close[i-1] <= r3_6h[i] and  # crossed above R3
                volume_confirmation and
                price > ema_34_1d_aligned[i]):  # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short breakdown: price crosses below S3 with volume, in downtrend
            elif (price < s3_6h[i] and close[i-1] >= s3_6h[i] and  # crossed below S3
                  volume_confirmation and
                  price < ema_34_1d_aligned[i]):  # downtrend filter
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below S3 (reversal) or trend fails
            if (price < s3_6h[i] and close[i-1] >= s3_6h[i]) or \
               price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above R3 (reversal) or trend fails
            if (price > r3_6h[i] and close[i-1] <= r3_6h[i]) or \
               price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0