#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation
# Long when Alligator Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when Alligator Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator alignment reverses (Lips crosses Teeth) or price crosses 1d EMA50
# Uses 12h primary timeframe with 1d HTF for trend filter (stable trend identification)
# Williams Alligator identifies trending vs ranging markets via smoothed moving averages
# Volume confirmation reduces false signals, trend filter ensures alignment with higher timeframe direction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Williams_Alligator_1dEMA50_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Smoothed Moving Averages (SMA with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA is similar to EMA but with different smoothing
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (Blue) - 13-period SMMA
    teeth = smma(close, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(close, 5)   # Lips (Green) - 5-period SMMA
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long conditions: bullish alignment AND price > Lips AND 1d close > 1d EMA50 AND volume spike
            if (bullish_alignment and 
                close[i] > lips[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment AND price < Lips AND 1d close < 1d EMA50 AND volume spike
            elif (bearish_alignment and 
                  close[i] < lips[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR price crosses below 1d EMA50 (trend reversal)
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR price crosses above 1d EMA50 (trend reversal)
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals