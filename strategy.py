#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Alligator Jaw = SMA(13,8), Teeth = SMA(8,5), Lips = SMA(5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator lines intertwine (Lips crosses Teeth or Jaw)
# Williams Alligator identifies trending vs ranging markets; effective in strong trends (bull/bear) and avoids whipsaws in ranges.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 4h (no HTF needed)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 4h
    # Jaw = SMA(13,8), Teeth = SMA(8,5), Lips = SMA(5,3)
    if len(close) >= 13:
        jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
        jaw = np.concatenate([np.full(8, np.nan), jaw[8:]])  # Shift by 8
        
        teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
        teeth = np.concatenate([np.full(5, np.nan), teeth[5:]])  # Shift by 5
        
        lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
        lips = np.concatenate([np.full(3, np.nan), lips[3:]])  # Shift by 3
    else:
        jaw = np.full(n, np.nan)
        teeth = np.full(n, np.nan)
        lips = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND volume filter AND above 1d EMA50
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                volume_filter[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND volume filter AND below 1d EMA50
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines intertwine (Lips crosses below Teeth or Jaw)
            if lips[i] < teeth[i] or lips[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines intertwine (Lips crosses above Teeth or Jaw)
            if lips[i] > teeth[i] or lips[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals