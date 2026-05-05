#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Alligator Jaw (13-period SMMA) AND price > EMA50(1w) AND volume > 2.0x 20-period average
# Short when price breaks below Alligator Jaw AND price < EMA50(1w) AND volume > 2.0x 20-period average
# Exit when price crosses back below/above Alligator Jaw OR trend flips (price crosses EMA50(1w))
# Williams Alligator uses smoothed moving averages (SMMA) with specific periods (13,8,5) and shifts (8,5,3)
# The Jaw (13-period, shifted 8) acts as a dynamic support/resistance level
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike confirms institutional participation
# Target: 7-25 trades/year per symbol (30-100 total over 4 years) for 1d timeframe
# Discrete sizing (0.25) to limit fee drag

name = "1d_WilliamsAlligator_JawBreak_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d data
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev_SMMA*(period-1) + Current_Price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close, 13)  # 13-period SMMA
    jaw = np.roll(jaw, 8)   # Shifted by 8 bars
    
    teeth = smma(close, 8)   # 8-period SMMA
    teeth = np.roll(teeth, 5) # Shifted by 5 bars
    
    lips = smma(close, 5)    # 5-period SMMA
    lips = np.roll(lips, 3)   # Shifted by 3 bars
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Alligator Jaw AND price > EMA50(1w) AND volume spike
            if (close[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Alligator Jaw AND price < EMA50(1w) AND volume spike
            elif (close[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Alligator Jaw (mean reversion) OR price < EMA50(1w) (trend flip)
            if (close[i] < jaw[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Alligator Jaw (mean reversion) OR price > EMA50(1w) (trend flip)
            if (close[i] > jaw[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals