#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d trend filter (price >/< EMA50) and volume spike (>2.0x median). Targets stronger institutional pivot levels (R3/S3) for higher conviction breakouts in trending markets. Uses discrete position sizing (0.25) to minimize fee churn. Designed for BTC/ETH with strict entry conditions (~15-25 trades/year) to avoid overtrading and work in bull/bear markets by only trading with 1d trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for Camarilla levels (HLC of prior 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    cam_high = pd.Series(df_12h['high'].values).shift(1).values
    cam_low = pd.Series(df_12h['low'].values).shift(1).values
    cam_close = pd.Series(df_12h['close'].values).shift(1).values
    
    # Camarilla R3, S3 levels (stronger breakout levels)
    R3 = cam_close + (cam_high - cam_low) * 1.1 / 4
    S3 = cam_close - (cam_high - cam_low) * 1.1 / 4
    
    # Volume spike filter: volume > 2.0x median volume (30-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d, Camarilla (need 2 bars for shift), volume median (30)
    start_idx = max(50, 2, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 2.0 * vol_median_val
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend
            long_signal = (close_val > r3_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S3 with volume spike, and downtrend
            short_signal = (close_val < s3_val) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
        elif position == -1:
            # Hold short
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0