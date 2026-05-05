#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Long when price breaks above 6h Camarilla R3 AND price > 12h EMA34 (uptrend) AND volume > 1.5x 20-period average
# Short when price breaks below 6h Camarilla S3 AND price < 12h EMA34 (downtrend) AND volume > 1.5x 20-period average
# Exit when price crosses 6h Camarilla pivot point (PP) OR EMA34 filter reverses
# Uses Camarilla levels for precise reversal/breakout levels, 12h EMA34 for regime filter
# Volume spike confirms institutional participation
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 6h (primary timeframe as required)
# Target: 75-150 total trades over 4 years (19-37/year) to minimize fee drag

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
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
    
    # Calculate 6h Camarilla levels (based on previous day's high/low/close)
    # Using previous 6h bar's high/low/close for current bar's levels
    prev_close = np.roll(close_6h, 1)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close[0] = close_6h[0]  # first bar uses current values
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    
    # Camarilla levels calculation
    # PP = (prev_high + prev_low + prev_close) / 3
    # R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    # S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 4.0
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4.0
    
    # Get 12h data ONCE before loop for EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    pp_aligned = align_htf_to_ltf(prices, df_6h, pp)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation on 6h (threshold: 1.5x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
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