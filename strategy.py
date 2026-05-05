#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
# Long when Williams %R < -80 (oversold) AND 1d close > 1d EMA50 AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND 1d close < 1d EMA50 AND volume > 2.0x 20-period average
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses 6h primary timeframe with 1d HTF for trend filter and Williams %R for mean reversion
# Williams %R identifies exhaustion points in trends; 1d EMA50 ensures alignment with higher-timeframe trend
# Volume spike confirms participation at extremes. Works in both bull and bear by fading extremes within trend.
# Target: 50-150 total trades over 4 years (12-37/year) based on proven mean reversion + trend filter edge

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R, EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R = -100 * (HH - C) / (HH - LL) where HH=highest high, LL=lowest low over lookback
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling highest high and lowest low for Williams %R (14-period)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    
    # Williams %R calculation
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average on 6h data
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND 1d close > 1d EMA50 AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND 1d close < 1d EMA50 AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (exhaustion ending)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (exhaustion ending)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals