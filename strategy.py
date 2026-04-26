#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and 1d volume confirmation.
Uses 4h/1d for signal direction (trend + volume regime) and 1h only for entry timing precision.
Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing (0.20) to minimize fee drag.
Session filter (08-20 UTC) avoids low-liquidity periods. Works in bull/bear via 4h trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = invalid
    trend_4h = np.where(ema_34_4h_aligned > 0, 
                        np.where(close > ema_34_4h_aligned, 1, -1), 
                        0)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d volume MA20 for regime filter
    volume_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20, adjust=False).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    # Volume regime: 1 = high volume (above MA), 0 = low/normal volume
    volume_regime = np.where(volume_ma_1d_aligned > 0, 
                             np.where(df_1d['volume'].values > volume_ma_1d_aligned, 1, 0), 
                             0)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # Calculate Camarilla pivot levels from 1d OHLC (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(trend_4h[i]) or np.isnan(volume_regime_aligned[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla R3/S3 breakout conditions with 4h trend and 1d volume confirmation
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 4h uptrend AND 1d high volume regime
            if close[i] > camarilla_r3_aligned[i] and trend_4h[i] == 1 and volume_regime_aligned[i] == 1:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 AND 4h downtrend AND 1d high volume regime
            elif close[i] < camarilla_s3_aligned[i] and trend_4h[i] == -1 and volume_regime_aligned[i] == 1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S3 OR 4h trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R3 OR 4h trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0