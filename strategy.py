#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla R3/S3 breakout with 1-week trend filter and volume confirmation.
In bull markets (price > 1w EMA50), long on R3 breakout with volume.
In bear markets (price < 1w EMA50), short on S3 breakout with volume.
Uses Camarilla levels for precise support/resistance, reducing false breakouts.
Target: 10-25 trades per year (~40-100 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels: R3/S3 = C +/- (H-L) * 1.1/2
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 2
    s3 = prev_close - camarilla_range * 1.1 / 2
    
    # Volume ratio: current volume / 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-week EMA50
        uptrend_regime = close[i] > ema_50_1w_aligned[i]
        downtrend_regime = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: close breaks above R3 in uptrend regime + volume
            long_entry = (close[i] > r3[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below S3 in downtrend regime + volume
            short_entry = (close[i] < s3[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below previous day's close or regime changes to downtrend
            if (close[i] < prev_close[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above previous day's close or regime changes to uptrend
            if (close[i] > prev_close[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals