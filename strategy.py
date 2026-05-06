#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Volatility Breakout with Volume Confirmation
# - Buy when price breaks above 1-day high with volume expansion
# - Sell when price breaks below 1-day low with volume expansion
# - Uses 1-day ATR to filter for high volatility breakouts only
# - Includes session filter (08-20 UTC) to avoid low liquidity periods
# - Designed to capture momentum bursts in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_1dVolatilityBreakout_VolumeConfirmation"
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
    
    # Get 1-day data for volatility breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day high and low from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate 1-day ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align 1-day levels to 6h timeframe
    prev_high_6h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_6h = align_htf_to_ltf(prices, df_1d, prev_low)
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (1.5 * vol_ma_20)  # Significant volume expansion
    volume_filter = volume > vol_ma_20  # Minimum volume threshold
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(prev_high_6h[i]) or np.isnan(prev_low_6h[i]) or np.isnan(atr_1d_6h[i]) or
            np.isnan(volume_expansion[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above previous day high with volume expansion
            # Only take if volatility is elevated (above average ATR)
            if (close[i] > prev_high_6h[i] and 
                volume_expansion[i] and 
                atr_1d_6h[i] > np.nanmedian(atr_1d_6h[max(0, i-50):i])):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below previous day low with volume expansion
            elif (close[i] < prev_low_6h[i] and 
                  volume_expansion[i] and 
                  atr_1d_6h[i] > np.nanmedian(atr_1d_6h[max(0, i-50):i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below previous day low (reversal) or time-based exit
            if close[i] < prev_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above previous day high (reversal) or time-based exit
            if close[i] > prev_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals