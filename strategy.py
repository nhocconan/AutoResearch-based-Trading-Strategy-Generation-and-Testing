#!/usr/bin/env python3
# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 (4h) AND close > 4h EMA50 AND volume > 1.8x average (1h)
# Short when price breaks below Camarilla S3 (4h) AND close < 4h EMA50 AND volume > 1.8x average (1h)
# Exit when price crosses Camarilla pivot point (mean reversion) OR trend reversal (price crosses 4h EMA50)
# Uses 1h timeframe with 4h trend filter to reduce noise. Target: 60-150 total trades over 4 years = 15-37/year.
# Camarilla pivots provide intraday structure; 4h EMA50 filters trend; volume confirms breakout.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivot calculation (HTF trend and structure)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (R3, S3, pivot) on previous 4h bar's OHLC to avoid look-ahead
    if len(high_4h) >= 1:
        # Use previous completed 4h bar
        prev_high = high_4h[-1] if len(high_4h) > 1 else high_4h[0]
        prev_low = low_4h[-1] if len(low_4h) > 1 else low_4h[0]
        prev_close = close_4h[-1] if len(close_4h) > 1 else close_4h[0]
        
        # Camarilla pivot calculation
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low
        r3 = pivot + (range_hl * 1.1 / 4)
        s3 = pivot - (range_hl * 1.1 / 4)
        
        # Create arrays of same length as 4h data
        camarilla_r3 = np.full_like(high_4h, r3)
        camarilla_s3 = np.full_like(high_4h, s3)
        camarilla_pivot = np.full_like(high_4h, pivot)
    else:
        camarilla_r3 = np.full_like(high_4h, np.nan)
        camarilla_s3 = np.full_like(high_4h, np.nan)
        camarilla_pivot = np.full_like(high_4h, np.nan)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Get 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current 1h volume > 1.8x 20-period average (spike confirmation)
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Camarilla R3 AND close > 4h EMA50 AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price < Camarilla S3 AND close < 4h EMA50 AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Camarilla pivot (mean reversion) OR trend reversal (close < 4h EMA50)
            if close[i] < camarilla_pivot_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price > Camarilla pivot (mean reversion) OR trend reversal (close > 4h EMA50)
            if close[i] > camarilla_pivot_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals