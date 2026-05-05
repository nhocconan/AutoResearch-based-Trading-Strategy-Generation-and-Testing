#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator Jaw (13-period SMMA) crosses above Teeth (8-period SMMA) AND price > Lips (5-period SMMA) 
# AND 1d close > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Jaw crosses below Teeth AND price < Lips AND 1d close < 1d EMA34 AND volume spike
# Exit when Jaw crosses Teeth in opposite direction (trend change)
# Uses 6h primary timeframe with 1d HTF for trend filter and Williams Alligator for entry timing
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-125 total trades over 4 years (19-31/year) based on proven Alligator effectiveness in trending markets
# Works in both bull and bear markets by following the 1d trend while using 6h Alligator for timing

name = "6h_Williams_Alligator_JawTeeth_Cross_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h data: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average - similar to EMA but with different smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + arr[i]) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Jaw crosses above Teeth AND price > Lips AND 1d close > 1d EMA34 AND volume spike
            if (jaw[i] > teeth[i] and jaw[i-1] <= teeth[i-1] and  # crossover
                close[i] > lips[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Jaw crosses below Teeth AND price < Lips AND 1d close < 1d EMA34 AND volume spike
            elif (jaw[i] < teeth[i] and jaw[i-1] >= teeth[i-1] and  # crossover
                  close[i] < lips[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Jaw crosses below Teeth (trend change to down)
            if jaw[i] < teeth[i] and jaw[i-1] >= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Jaw crosses above Teeth (trend change to up)
            if jaw[i] > teeth[i] and jaw[i-1] <= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals