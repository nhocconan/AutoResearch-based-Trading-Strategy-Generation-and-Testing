#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND price > EMA50(1d) AND volume > 2.0x 20-period average
# Short when Jaw > Teeth > Lips (bearish alignment) AND price < EMA50(1d) AND volume > 2.0x 20-period average
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips not monotonic) OR trend flips
# Uses discrete sizing (0.30) to limit fee drag. Target: 12-30 trades/year per symbol.
# Williams Alligator identifies trend absence/presence via smoothed medians, effective in both bull (longs in uptrend+ bull alignment) and bear (shorts in downtrend+ bear alignment) markets.
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws, volume spike confirms institutional participation.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
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
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator on 12h data (Smoothed Medians)
    # Jaw: Smoothed Median (13 periods, 8 bars ahead)
    # Teeth: Smoothed Median (8 periods, 5 bars ahead) 
    # Lips: Smoothed Median (5 periods, 3 bars ahead)
    median = (high + low) / 2.0
    jaw = pd.Series(median).rolling(window=13, min_periods=13).median().shift(8).values
    teeth = pd.Series(median).rolling(window=8, min_periods=8).median().shift(5).values
    lips = pd.Series(median).rolling(window=5, min_periods=5).median().shift(3).values
    
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
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Jaw < Teeth < Lips (bullish alignment) AND price > EMA50(1d) AND volume spike
            if (jaw[i] < teeth[i] and 
                teeth[i] < lips[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Jaw > Teeth > Lips (bearish alignment) AND price < EMA50(1d) AND volume spike
            elif (jaw[i] > teeth[i] and 
                  teeth[i] > lips[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (not Jaw < Teeth < Lips) OR price < EMA50(1d) (trend flip)
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or \
               close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Alligator alignment breaks (not Jaw > Teeth > Lips) OR price > EMA50(1d) (trend flip)
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or \
               close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals