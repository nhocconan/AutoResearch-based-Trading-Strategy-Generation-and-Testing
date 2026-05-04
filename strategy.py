#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance; R3/S3 are strong reversal levels
# Breakout above R3 with uptrend EMA34 and volume spike = long entry
# Breakdown below S3 with downtrend EMA34 and volume spike = short entry
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# Discrete sizing 0.25 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4, etc.
    # We only need R3 and S3 for breakout signals
    camarilla_r3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above Camarilla R3 AND 1d EMA34 uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below Camarilla S3 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla R3 OR below EMA13 (using 4h EMA13 for dynamic stop)
            # Calculate 4h EMA13 for dynamic exit
            if i >= 13:
                close_s = pd.Series(close[:i+1])
                ema13_4h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().iloc[-1]
                if close[i] < camarilla_r3_aligned[i] or close[i] < ema13_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla S3 OR above EMA13 (using 4h EMA13 for dynamic stop)
            if i >= 13:
                close_s = pd.Series(close[:i+1])
                ema13_4h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().iloc[-1]
                if close[i] > camarilla_s3_aligned[i] or close[i] > ema13_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals