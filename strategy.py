#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 1d close > EMA34 and volume > 2x 20-period MA.
# Short when price breaks below S3 with 1d close < EMA34 and volume > 2x 20-period MA.
# Exit when price reverts to the 1d EMA34 level or volume drops below average.
# Uses 6h timeframe for 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide precise intraday support/resistance, EMA34 filters trend direction,
# volume spike confirms institutional participation. Designed to work in both bull and bear markets
# by trading breakouts in the direction of the higher timeframe trend.

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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align 1d indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 6h volume > 2x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 2.0)
        
        if position == 0:
            # Long: price breaks above R3 AND 1d close > EMA34 AND volume spike AND session
            if close[i] > camarilla_r3_aligned[i] and close_1d[-1] > ema_34_1d[-1] and volume_spike:
                # Check if we have valid 1d close for today (use previous day's close for signal)
                # Since we're using aligned arrays, we need to check the 1d close that corresponds to this 6h bar
                # For simplicity, we use the EMA alignment to infer trend
                if close[i] > camarilla_r3_aligned[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 AND 1d close < EMA34 AND volume spike AND session
            elif close[i] < camarilla_s3_aligned[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to EMA34 OR volume drops below average
            if close[i] <= ema_34_1d_aligned[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to EMA34 OR volume drops below average
            if close[i] >= ema_34_1d_aligned[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals