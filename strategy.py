#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (SMMA-based) with volume confirmation and 1w trend filter.
# Uses SMMA (Smoothed Moving Average) to filter noise and identify trend direction.
# Long when price > Alligator Jaw (13-period SMMA) and Lips > Teeth (bullish alignment) with volume confirmation.
# Short when price < Jaw and Lips < Teeth (bearish alignment) with volume confirmation.
# 1w EMA trend filter ensures alignment with higher timeframe trend.
# Designed to work in both bull and bear markets by following 1w EMA direction.
# Williams Alligator is effective in trending markets and avoids whipsaw in ranges.
name = "1d_WilliamsAlligator_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing"""
    if len(source) < length:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (prev_SMMA * (length-1) + current_price) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator components (13, 8, 5 periods SMMA)
    # Jaw: 13-period SMMA
    jaw = smma(close, 13)
    # Teeth: 8-period SMMA
    teeth = smma(close, 8)
    # Lips: 5-period SMMA
    lips = smma(close, 5)
    
    # 1w EMA trend filter (50-period)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > Jaw AND Lips > Teeth (bullish alignment) + volume + 1w EMA up
            if (price > jaw[i] and lips[i] > teeth[i] and vol_confirm[i] and price > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND Lips < Teeth (bearish alignment) + volume + 1w EMA down
            elif (price < jaw[i] and lips[i] < teeth[i] and vol_confirm[i] and price < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Jaw or Lips < Teeth (loss of bullish alignment)
            if price < jaw[i] or lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Jaw or Lips > Teeth (loss of bearish alignment)
            if price > jaw[i] or lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals