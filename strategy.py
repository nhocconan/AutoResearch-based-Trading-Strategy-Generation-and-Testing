#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 12h trend filter and volume confirmation
# Long when price breaks above R4 AND 12h close > 12h EMA34 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below S4 AND 12h close < 12h EMA34 (downtrend) AND volume > 1.5x 20 EMA
# Uses 6h for entry timing, 12h for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "6h_Camarilla_R4S4_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels (based on previous 12h's OHLC)
    # We need 12h OHLC for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 12h OHLC arrays
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # Calculate Camarilla levels for each 12h period
    # R4 = close + (high - low) * 1.1
    # S4 = close - (high - low) * 1.1
    camarilla_r4 = close_12h + (high_12h - low_12h) * 1.1
    camarilla_s4 = close_12h - (high_12h - low_12h) * 1.1
    
    # Align 12h Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Get 12h data for trend filter - ONCE before loop
    close_12h_vals = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h_vals).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_12h = close_12h_vals > ema_34_12h
    downtrend_12h = close_12h_vals < ema_34_12h
    
    # Align 12h trend to 6h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4 AND 12h uptrend AND volume spike
            if (close[i] > r4_aligned[i] and 
                uptrend_12h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4 AND 12h downtrend AND volume spike
            elif (close[i] < s4_aligned[i] and 
                  downtrend_12h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S4 OR 12h trend changes to downtrend
            if (close[i] < s4_aligned[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R4 OR 12h trend changes to uptrend
            if (close[i] > r4_aligned[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals