#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND close > EMA50(1d) AND volume > 1.8x 20-period average
# Short when Jaw > Teeth > Lips (bearish alignment) AND close < EMA50(1d) AND volume > 1.8x 20-period average
# Exit when Alligator alignment breaks OR EMA50(1d) trend flips
# Uses 12h primary timeframe with 1d HTF for trend filter to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Works in both bull and bear markets by adapting to trend via EMA filter

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate SMMA lines
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift according to Alligator specification
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars ahead
    teeth = np.roll(teeth, 5)  # Teeth shifted 5 bars ahead
    lips = np.roll(lips, 3)    # Lips shifted 3 bars ahead
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish alignment (Jaw < Teeth < Lips) AND close > EMA50(1d) AND volume spike
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish alignment (Jaw > Teeth > Lips) AND close < EMA50(1d) AND volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bullish alignment breaks OR close < EMA50(1d) (trend flip)
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bearish alignment breaks OR close > EMA50(1d) (trend flip)
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals