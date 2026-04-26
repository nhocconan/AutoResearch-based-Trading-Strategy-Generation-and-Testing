#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeRegime
Hypothesis: Use 4h for primary trend (EMA50) and 1d for volume regime (top 30% volume days), then 1h for precise entry timing on Camarilla R3/S3 breakouts. This combines HTF directional alignment with volume confirmation while using lower timeframe only for entry precision to minimize trades. Fixed size 0.20 to control fees. Designed to work in both bull (trend filters) and bear (mean reversion exits) markets.
"""

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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h_vals + (rng_4h * 1.1 / 4)   # R3 level
    camarilla_s3_4h = close_4h_vals - (rng_4h * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_1d = df_1d['volume'].values
    vol_series_1d = pd.Series(vol_1d)
    vol_percentile_70_1d = vol_series_1d.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime_1d = vol_1d > vol_percentile_70_1d
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d.astype(float))
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to control trade frequency (0.20 = 20%)
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and volume percentile)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_regime_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_regime = volume_regime_aligned[i] > 0.5  # convert back to boolean
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume regime AND 4h EMA50 trend filter
        long_entry = (close_val > camarilla_r3_val) and vol_regime and (close_val > ema_50_val)
        short_entry = (close_val < camarilla_s3_val) and vol_regime and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeRegime"
timeframe = "1h"
leverage = 1.0