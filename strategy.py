#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses Camarilla R3/S3 levels from daily pivot for breakout entries on 12h chart.
# Enters long when price breaks above R3 with volume spike and daily uptrend (close > EMA34).
# Enters short when price breaks below S3 with volume spike and daily downtrend (close < EMA34).
# Exits when price re-enters the Camarilla range (between S3 and R3) or volatility drops.
# Uses volume confirmation (1.5x average volume) and ATR filter to avoid false breakouts.
# Designed for 12h timeframe targeting 15-35 trades per year with position size 0.25.

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
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate average volume for volume spike detection
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * avg_volume[i]
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3_aligned[i]
        breakout_short = close[i] < camarilla_s3_aligned[i]
        
        # Re-entry conditions (exit when price returns to Camarilla range)
        reentry_long = close[i] < camarilla_r3_aligned[i]  # Exit long when price drops below R3
        reentry_short = close[i] > camarilla_s3_aligned[i]  # Exit short when price rises above S3
        
        if position == 0:
            # Long entry: breakout above R3 with volume spike and daily uptrend
            if (breakout_long and 
                volume_spike and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below S3 with volume spike and daily downtrend
            elif (breakout_short and 
                  volume_spike and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters Camarilla range (below R3) or volatility drops
            if (reentry_long or 
                atr[i] < 0.5 * atr[i-1]):  # Volatility drop filter
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Camarilla range (above S3) or volatility drops
            if (reentry_short or 
                atr[i] < 0.5 * atr[i-1]):  # Volatility drop filter
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals