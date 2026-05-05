#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and session filter (08-20 UTC)
# Long when price breaks above Camarilla R3 AND 4h close > 4h EMA50 AND session 08-20 UTC
# Short when price breaks below Camarilla S3 AND 4h close < 4h EMA50 AND session 08-20 UTC
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-30 trades/year per symbol.
# Camarilla provides intraday structure; 4h EMA50 filters medium-term trend; session filter avoids Asian session noise.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# 1h timeframe with tight entry conditions balances trade frequency and execution precision.

name = "1h_Camarilla_R3S3_4hEMA50_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 1h data ONCE before loop for Camarilla calculation (based on previous bar)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 5:
        return np.zeros(n)
    
    # Calculate previous 1h bar's OHLC for Camarilla levels
    prev_close = df_1h['close'].shift(1).values
    prev_high = df_1h['high'].shift(1).values
    prev_low = df_1h['low'].shift(1).values
    
    # Camarilla levels for previous 1h bar
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (already aligned since same TF)
    # No need to align as we're using 1h data for 1h timeframe
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_4h = close_4h > ema_50_4h
    downtrend_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any value is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > R3 AND 4h uptrend AND in session
            if (close[i] > R3[i] and 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short conditions: price < S3 AND 4h downtrend AND in session
            elif (close[i] < S3[i] and 
                  downtrend_4h_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < S3 OR 4h trend changes to downtrend
            if (close[i] < S3[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > R3 OR 4h trend changes to uptrend
            if (close[i] > R3[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals