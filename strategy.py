#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Long at S3 (1.118/6 range) when price > 1d EMA(50) and volume > 1.5x avg
# Short at R3 (1.118/6 range) when price < 1d EMA(50) and volume > 1.5x avg
# Exit at opposite S1/R1 level or opposite Camarilla extreme
# Camarilla levels provide intraday support/resistance that work in ranging markets
# 1d EMA filter avoids counter-trend trades, volume confirms breakouts
# Target: 50-150 trades over 4 years (12-37/year) with controlled risk

name = "6h_camarilla_reversal_1dema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Previous day's typical price for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_open = np.roll(open_1d, 1)
    
    # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_open[0] = open_1d[0]
    
    # Camarilla pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels
    s1 = close_1d - (range_ * 1.1 / 12)
    s2 = close_1d - (range_ * 1.1 / 6)
    s3 = close_1d - (range_ * 1.1 / 4)
    r1 = close_1d + (range_ * 1.1 / 12)
    r2 = close_1d + (range_ * 1.1 / 6)
    r3 = close_1d + (range_ * 1.1 / 4)
    s4 = close_1d - (range_ * 1.1 / 2)
    r4 = close_1d + (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    
    # 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(s3_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(r1_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price reaches S1 (take profit) or breaks below S4 (stop)
            if close[i] <= s1_6h[i] or close[i] < s4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R1 (take profit) or breaks above R4 (stop)
            if close[i] >= r1_6h[i] or close[i] > r4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla reversal + trend filter + volume
            if volume[i] > volume_threshold[i]:
                # Long at S3 in uptrend
                if close[i] <= s3_6h[i] and close[i] > ema_50_6h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at R3 in downtrend
                elif close[i] >= r3_6h[i] and close[i] < ema_50_6h[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals