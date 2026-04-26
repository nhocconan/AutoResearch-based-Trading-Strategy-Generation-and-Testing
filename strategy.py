#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 in 1d uptrend with volume spike (>2x 20-bar MA).
Short when price breaks below S3 in 1d downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year on 4h.
Works in bull/bear by following 1d trend. Camarilla levels provide intraday support/resistance.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate Camarilla levels from previous day's OHLC
    # We need to get the previous day's high, low, close for each 4h bar
    # Since we're on 4h timeframe, we'll use the 1d data shifted by 1
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Align previous day's OHLC to 4h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    camarilla_range = prev_day_high_aligned - prev_day_low_aligned
    r3 = prev_day_close_aligned + camarilla_range * 1.1 / 2
    s3 = prev_day_close_aligned - camarilla_range * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 1 for previous day)
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(prev_day_high_aligned[i]) or np.isnan(prev_day_low_aligned[i]) or
            np.isnan(prev_day_close_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1d uptrend and volume spike
            if (close[i] > r3[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1d downtrend and volume spike
            elif (close[i] < s3[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S3 OR 1d trend changes to downtrend
            if (close[i] < s3[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R3 OR 1d trend changes to uptrend
            if (close[i] > r3[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0