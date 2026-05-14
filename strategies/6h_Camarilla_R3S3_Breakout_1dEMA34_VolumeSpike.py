#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above 6h Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume > 2x 20-period average
# Short when price breaks below 6h Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume > 2x 20-period average
# Exit when price crosses 6h Camarilla pivot point (PP) OR EMA34 filter reverses
# Uses Camarilla levels for precise intraday structure, 1d EMA34 for trend regime (avoid whipsaws)
# Volume spike confirms institutional breakout participation
# Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
# Timeframe: 6h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Camarilla levels from previous 6h bar
    # PP = (high + low + close) / 3
    # R3 = PP + (high - low) * 1.1/2
    # S3 = PP - (high - low) * 1.1/2
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h = np.roll(close_6h, 1)
    prev_high_6h[0] = high_6h[0]
    prev_low_6h[0] = low_6h[0]
    prev_close_6h[0] = close_6h[0]
    
    pp_6h = (prev_high_6h + prev_low_6h + prev_close_6h) / 3.0
    r3_6h = pp_6h + (prev_high_6h - prev_low_6h) * 1.1 / 2.0
    s3_6h = pp_6h - (prev_high_6h - prev_low_6h) * 1.1 / 2.0
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    pp_aligned = align_htf_to_ltf(prices, df_6h, pp_6h)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below PP OR price < EMA34 (trend weakening)
            if close[i] < pp_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above PP OR price > EMA34 (trend weakening)
            if close[i] > pp_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals