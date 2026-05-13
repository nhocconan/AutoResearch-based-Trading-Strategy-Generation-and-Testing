#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (>1.8x avg volume).
# Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Camarilla levels provide strong intraday support/resistance. EMA50 on 12h filters counter-trend trades.
# Volume confirmation ensures breakouts have participation. Works in bull via longs, bear via shorts with trend filter.

name = "4h_Camarilla_R3_S3_Breakout_12hEMATrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (wait for 12h bar to close)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla pivot calculation (using 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for R3, S3, R4, S4
    # Camarilla formula: Range = (high - low)
    # R4 = close + Range * 1.1/2
    # R3 = close + Range * 1.1/4
    # S3 = close - Range * 1.1/4
    # S4 = close - Range * 1.1/2
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + daily_range * 1.1 / 4
    camarilla_s3 = close_1d - daily_range * 1.1 / 4
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND 12h EMA50 rising AND volume > 1.8x average
            # Use R4 as stronger breakout confirmation
            if (close[i] > camarilla_r4_aligned[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S3 AND 12h EMA50 falling AND volume > 1.8x average
            # Use S4 as stronger breakdown confirmation
            elif (close[i] < camarilla_s4_aligned[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR) or re-entry below R3
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            reentry_signal = close[i] < camarilla_r3_aligned[i]
            if trailing_stop or reentry_signal:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR) or re-entry above S3
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            reentry_signal = close[i] > camarilla_s3_aligned[i]
            if trailing_stop or reentry_signal:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals