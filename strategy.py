#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 12h Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as strong support/resistance on 6h timeframe. 
Price rejection at R3/S3 with reversal signals (engulfing candle) + 12h EMA trend filter avoids counter-trend trades. 
Works in bull/bear by aligning with higher timeframe trend. Targets 20-40 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_12h_trend_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = df_12h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous period's high, low, close for Camarilla calculation
    # Using 12h data to calculate pivots for 6b timeframe (2 bars = 12h)
    prev_high = df_12h['high'].values
    prev_low = df_12h['low'].values
    prev_close = df_12h['close'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # Range = previous high - previous low
    rng = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + rng * 1.1 / 2
    r3 = prev_close + rng * 1.1 / 4
    s3 = prev_close - rng * 1.1 / 4
    s4 = prev_close - rng * 1.1 / 2
    
    # Align to 6h timeframe (each 12h bar corresponds to 2x 6h bars)
    r4_6h = np.repeat(r4, 2)
    r3_6h = np.repeat(r3, 2)
    s3_6h = np.repeat(s3, 2)
    s4_6h = np.repeat(s4, 2)
    
    # Trim/pad to match 6h length
    min_len = min(len(r4_6h), n)
    r4_6h = r4_6h[:min_len]
    r3_6h = r3_6h[:min_len]
    s3_6h = s3_6h[:min_len]
    s4_6h = s4_6h[:min_len]
    
    # Pad if shorter
    if len(r4_6h) < n:
        pad_len = n - len(r4_6h)
        r4_6h = np.concatenate([r4_6h, np.full(pad_len, np.nan)])
        r3_6h = np.concatenate([r3_6h, np.full(pad_len, np.nan)])
        s3_6h = np.concatenate([s3_6h, np.full(pad_len, np.nan)])
        s4_6h = np.concatenate([s4_6h, np.full(pad_len, np.nan)])
    
    # Engulfing candle detection
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)
    # Actually: current candle engulfs previous
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)
    # Correct implementation:
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > np.roll(open_price, 1)) & (open_price < np.roll(close, 1))
    bearish_engulf = (close < open_price) & (open_price > close) & \
                     (close < np.roll(open_price, 1)) & (open_price > np.roll(close, 1))
    
    # Handle first bar
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 OR bearish engulf at resistance
            if (close[i] <= s3_6h[i] or bearish_engulf[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 OR bullish engulf at support
            if (close[i] >= r3_6h[i] or bullish_engulf[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price at S3/S4 support with bullish engulf, uptrend
            if ((low[i] <= s3_6h[i] * 1.005 or low[i] <= s4_6h[i] * 1.005) and 
                bullish_engulf[i] and 
                close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price at R3/R4 resistance with bearish engulf, downtrend
            elif ((high[i] >= r3_6h[i] * 0.995 or high[i] >= r4_6h[i] * 0.995) and 
                  bearish_engulf[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals