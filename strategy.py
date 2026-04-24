#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (R3/S3) from prior completed 1d candles for breakout detection.
- Breakout above R3 or below S3 with volume > 2.0x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1d EMA34 to align with higher timeframe direction.
- Designed for 4h timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 25-50 trades/year (100-200 total over 4 years) to stay fee-efficient.
- Based on proven pattern: Camarilla breakout + volume + trend filter showed strong performance in DB.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior completed 1d close for EMA34
    close_1d = df_1d['close'].shift(1).values
    
    # 1d EMA34 trend filter
    close_1d_series = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d_series).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels (R3, S3) from prior completed 1d candle
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d_shift = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    camarilla_range = high_1d - low_1d
    camarilla_r3 = close_1d_shift + camarilla_range * 1.1 / 4
    camarilla_s3 = close_1d_shift - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to LTF
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Camarilla R3 AND price above 1d EMA34 AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Camarilla S3 AND price below 1d EMA34 AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Camarilla S3 OR price below 1d EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Camarilla R3 OR price above 1d EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0