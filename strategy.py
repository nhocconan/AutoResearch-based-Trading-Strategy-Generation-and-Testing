#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 AND 1d EMA34 > EMA89 (bullish trend) AND 4h volume > 2x 20-period average.
# Short when price breaks below S3 AND 1d EMA34 < EMA89 (bearish trend) AND 4h volume > 2x 20-period average.
# Exit when price crosses back below R3 (for long) or above S3 (for short).
# Uses proven Camarilla structure with trend filter to avoid false breakouts in ranging markets.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "4h_Camarilla_R3S3_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Note: For intraday calculation, we use previous day's OHLC
    # For 4h timeframe, we calculate daily levels and propagate forward
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # Handle first value
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_close - (range_ * 1.1 / 4)
    
    # 4h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1d data for trend filter (EMA34 and EMA89)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Trend: bullish when EMA34 > EMA89, bearish when EMA34 < EMA89
    trend_bullish = ema34_1d > ema89_1d
    trend_bearish = ema34_1d < ema89_1d
    
    # Align 1d trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 90  # Sufficient warmup for EMA89
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
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
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below R3
            if close[i] < camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above S3
            if close[i] > camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals