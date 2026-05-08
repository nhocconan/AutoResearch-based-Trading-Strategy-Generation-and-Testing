#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
# Long when price breaks above R3 AND 4h EMA20 > EMA50 (bullish trend) AND volume > 1.3x 20-period average.
# Short when price breaks below S3 AND 4h EMA20 < EMA50 (bearish trend) AND volume > 1.3x 20-period average.
# Exit when price crosses back below R1 (for long) or above S1 (for short).
# Uses Camarilla levels for precise intraday support/resistance with trend filter to avoid false breakouts.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "1h_Camarilla_R3S3_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels for today
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_close - (range_ * 1.1 / 4)
    camarilla_r1 = prev_close + (range_ * 1.1 / 12)
    camarilla_s1 = prev_close - (range_ * 1.1 / 12)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    # 4h data for trend filter (EMA20 > EMA50 = bullish)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish = ema20_4h > ema50_4h
    trend_bearish = ema20_4h < ema50_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, bullish trend, volume spike
            long_cond = (close[i] > camarilla_r3[i]) and trend_bullish_aligned[i] and volume_filter[i]
            # Short conditions: break below S3, bearish trend, volume spike
            short_cond = (close[i] < camarilla_s3[i]) and trend_bearish_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: cross below R1
            if close[i] < camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: cross above S1
            if close[i] > camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals