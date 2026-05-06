#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band with price > 12h EMA50 and volume > 1.5x average
# Short when price breaks below Donchian lower band with price < 12h EMA50 and volume > 1.5x average
# Uses 4h Donchian channels for breakout signals, 12h EMA50 for trend filter, volume for confirmation
# Designed to work in bull markets via breakouts above resistance and in bear markets via breakdowns below support
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_Donchian20_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA50 on 4h close (needs 50 bars)
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 12h Donchian Channel (20-period high/low)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 20-period high and low
    high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_donchian = align_htf_to_ltf(prices, df_12h, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with uptrend and volume confirmation
            if close[i] > upper_donchian[i] and close[i] > ema50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian with downtrend and volume confirmation
            elif close[i] < lower_donchian[i] and close[i] < ema50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals