#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume spike confirmation
# Uses daily Camarilla levels (S3, S2, R2, R3) for mean reversion entries in ranging markets.
# Only takes long positions when price > 1d EMA34 (uptrend filter) and short when price < 1d EMA34 (downtrend filter).
# Volume > 2x 20-period average confirms reversal strength.
# Target: 20-30 trades/year to minimize fee decay while capturing high-probability reversals.
# Works in both bull and bear markets by aligning with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for previous day
    camarilla_S3 = np.full(n_1d, np.nan)
    camarilla_S2 = np.full(n_1d, np.nan)
    camarilla_R2 = np.full(n_1d, np.nan)
    camarilla_R3 = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        # Camarilla formula using previous day's range
        # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4
        # S3 = C - (H-L)*1.1/4, S2 = C - (H-L)*1.1/2
        # We use S3/S2 for long entries, R2/R3 for short entries
        camarilla_S3[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 4
        camarilla_S2[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 2
        camarilla_R2[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 2
        camarilla_R3[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 4
    
    # Align daily indicators to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S2)
    camarilla_R2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R2)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = vol_period
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_S2_aligned[i]) or
            np.isnan(camarilla_R2_aligned[i]) or
            np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: price touches S3/S2 with uptrend filter and volume spike
            long_condition = (
                (price <= camarilla_S3_aligned[i] * 1.005 or price <= camarilla_S2_aligned[i] * 1.005) and
                price > ema_34_1d_aligned[i] and
                vol_ratio > 2.0
            )
            # Short: price touches R2/R3 with downtrend filter and volume spike
            short_condition = (
                (price >= camarilla_R3_aligned[i] * 0.995 or price >= camarilla_R2_aligned[i] * 0.995) and
                price < ema_34_1d_aligned[i] and
                vol_ratio > 2.0
            )
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to EMA34 or reaches Camarilla R3
            if price >= ema_34_1d_aligned[i] or price >= camarilla_R3_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to EMA34 or reaches Camarilla S3
            if price <= ema_34_1d_aligned[i] or price <= camarilla_S3_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Reversal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0