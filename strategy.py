#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike
# Uses 12h Camarilla pivot levels (R3/S3) for structural breakouts
# Enters long when price breaks above 12h R3 + volume > 1.8 x 20-period EMA + bullish 1d EMA34 trend
# Enters short when price breaks below 12h S3 + volume > 1.8 x 20-period EMA + bearish 1d EMA34 trend
# Exits on opposite Camarilla breakout or when 1d trend reverses
# Volume spike confirms institutional participation, reducing false breakouts
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Camarilla levels work in both bull/bear markets as dynamic support/resistance

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Camarilla pivot levels (R3, S3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: Range = High - Low
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    range_12h = high_12h - low_12h
    r3_level = close_12h + (range_12h * 1.1 / 4)
    s3_level = close_12h - (range_12h * 1.1 / 4)
    
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_level)
    
    # Get 12h data for volume EMA(20) for volume confirmation
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume EMA(20) for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.8 x 20-period EMA
        volume_confirmed = volume[i] > (1.8 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume confirmation + bullish 1d trend
            if (close[i] > r3_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume confirmation + bearish 1d trend
            elif (close[i] < s3_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 OR 1d trend turns bearish
            if close[i] < s3_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 OR 1d trend turns bullish
            if close[i] > r3_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals