#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 4h Candlestick pattern (Engulfing) at daily pivot levels with volume confirmation
    # Uses Engulfing patterns at S3/R3 levels for high-probability reversals
    # Works in both bull and bear markets by capturing reversals at key support/resistance
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot range (high-low)
    range_1d = high_1d - low_1d
    close_prev = close_1d
    
    # Daily S3 and R3 levels
    s3_1d = close_prev - (range_1d * 3.0 / 6)
    r3_1d = close_prev + (range_1d * 3.0 / 6)
    
    # Align daily S3/R3 to 4h
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # 4h data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bullish Engulfing: current green candle fully engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < np.roll(close, 1)) & \
                     (np.roll(close, 1) < np.roll(open_price, 1))
    # Bearish Engulfing: current red candle fully engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price > close) & \
                     (close < open_price) & (open_price > np.roll(close, 1)) & \
                     (np.roll(close, 1) > np.roll(open_price, 1))
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish engulfing at or below S3 with volume surge
            if bullish_engulf[i] and low[i] <= s3_1d_aligned[i] * 1.002 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing at or above R3 with volume surge
            elif bearish_engulf[i] and high[i] >= r3_1d_aligned[i] * 0.998 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses the opposite S3/R3 level
            if position == 1:
                if high[i] >= r3_1d_aligned[i] * 0.998:  # Reached R3, take profit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if low[i] <= s3_1d_aligned[i] * 1.002:  # Reached S3, take profit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Engulfing_S3_R3_Pivot_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0