#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d HTF trend filter
# Uses Williams Alligator (Smoothed Medians: Jaw=13, Teeth=8, Lips=5) on 12h for trend direction
# 1d close > 1d EMA50 for bullish regime filter (avoid shorts in strong uptrends, avoid longs in strong downtrends)
# Volume confirmation: 12h volume > 1.5x 20-period average to filter low-quality breakouts
# Long when Alligator is bullish (Lips > Teeth > Jaw) AND 1d close > 1d EMA50 AND volume spike
# Short when Alligator is bearish (Lips < Teeth < Jaw) AND 1d close < 1d EMA50 AND volume spike
# Exit when Alligator direction reverses (Lips crosses Teeth)
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Williams Alligator identifies trending vs ranging markets; 1d EMA50 filters for higher-timeframe regime; volume confirms participation

name = "12h_WilliamsAlligator_BullBear_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate Williams Alligator on 12h data
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Smoothed medians (SMMA) - using EMA as approximation for SMMA with same period
    # Jaw: 13-period, Teeth: 8-period, Lips: 5-period
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
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
            # Long conditions: Alligator bullish (Lips > Teeth > Jaw) AND 1d close > 1d EMA50 AND volume spike
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish (Lips < Teeth < Jaw) AND 1d close < 1d EMA50 AND volume spike
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish (Lips crosses below Teeth)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish (Lips crosses above Teeth)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals