#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Breakout of Camarilla R3/S3 levels on 12h chart with 1-day trend filter (close > EMA34 for long, < EMA34 for short) and volume spike confirmation (volume > 1.5x 20-period average). Designed to work in both bull and bear markets by following the higher timeframe trend. Targets 12-37 trades per year on 12h timeframe with position size 0.25.

name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average (20-period) for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 12h bar
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # Using previous bar's high/low/close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 2
    s3 = prev_close - camarilla_range * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume spike: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_above_r3 = close[i] > r3[i]
        breakout_below_s3 = close[i] < s3[i]
        
        if position == 0:
            # Long entry: breakout above R3 with uptrend and volume spike
            if (breakout_above_r3 and 
                price_above_ema and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below S3 with downtrend and volume spike
            elif (breakout_below_s3 and 
                  price_below_ema and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below previous day's close (mean reversion) or trend fails
            if (close[i] < prev_close[i] or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above previous day's close or trend fails
            if (close[i] > prev_close[i] or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals