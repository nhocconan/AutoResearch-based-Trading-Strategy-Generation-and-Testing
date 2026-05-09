#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with daily EMA34 trend and volume confirmation
# Designed for low trade frequency (12-37/year) to avoid fee drag.
# Camarilla levels from 1d provide strong intraday support/resistance.
# EMA34 filter ensures alignment with daily trend. Volume confirms breakout strength.
# Works in bull/bear markets by requiring alignment with daily trend.
name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot Point = Typical Price
    pivot = typical_price.values
    # R3 = H + 2*(P - L)
    r3 = df_1d['high'].values + 2 * (pivot - df_1d['low'].values)
    # S3 = L - 2*(H - P)
    s3 = df_1d['low'].values - 2 * (df_1d['high'].values - pivot)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        bullish_breakout = close[i] > r3_12h[i]  # Break above R3
        bearish_breakout = close[i] < s3_12h[i]  # Break below S3
        trend_up = close[i] > ema_34_12h[i]
        trend_down = close[i] < ema_34_12h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if bullish_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif bearish_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or trend reversal
            if bearish_breakout or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout or trend reversal
            if bullish_breakout or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals