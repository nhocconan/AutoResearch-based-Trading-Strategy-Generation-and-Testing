#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# In trending markets (4h close > 20 EMA), trade breakouts in trend direction: long on R3 breakout, short on S3 breakdown
# In ranging markets (4h close near 20 EMA), fade extremes: short near R3, long near S3
# Volume confirmation (>1.3x 20-period EMA) reduces false signals. Uses 1h timeframe targeting 80-120 trades over 4 years.
# Discrete position sizing (0.20) minimizes fee churn and manages drawdown in both bull and bear markets.

name = "1h_Camarilla_R3S3_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # Camarilla equations: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use daily OHLC to calculate Camarilla levels for 1h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_width = (daily_high - daily_low) * 1.1
    camarilla_r3 = daily_close + camarilla_width / 4
    camarilla_s3 = daily_close - camarilla_width / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(ema_20_4h[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine market state based on 4h EMA
            # Trending: price significantly above/below EMA
            # Ranging: price near EMA
            ema_distance = abs(close[i] - ema_20_4h[i]) / ema_20_4h[i]
            
            if ema_distance > 0.015:  # Trending market (>1.5% deviation from EMA)
                # Trade breakouts in trend direction
                if close[i] > r3_aligned[i] and volume_confirm:
                    signals[i] = 0.20
                    position = 1
                elif close[i] < s3_aligned[i] and volume_confirm:
                    signals[i] = -0.20
                    position = -1
            else:  # Ranging market (near EMA)
                # Fade extremes at Camarilla levels
                if close[i] <= s3_aligned[i] and volume_confirm:
                    signals[i] = 0.20
                    position = 1
                elif close[i] >= r3_aligned[i] and volume_confirm:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price retouches opposite Camarilla level OR volume drops
            if (close[i] >= r3_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retouches opposite Camarilla level OR volume drops
            if (close[i] <= s3_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals