#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above Alligator Jaw (blue line) AND 1w close > 1w EMA34 AND volume > 2x 20-period average
# Short when price breaks below Alligator Jaw (blue line) AND 1w close < 1w EMA34 AND volume > 2x 20-period average
# Exit when price crosses back below/above Alligator Jaw
# Uses 1d primary timeframe with 1w HTF for trend filter and Alligator structure
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) - we use Jaw as the main trend line
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) based on proven Alligator effectiveness in trending markets
# Works in both bull and bear markets by following the 1w trend while using 1d for entry timing

name = "1d_Williams_Alligator_Jaw_Breakout_1wEMA34_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 1d data
    # Jaw (blue line): 13-period SMMA smoothed by 8 periods
    # Teeth (red line): 8-period SMMA smoothed by 5 periods
    # Lips (green line): 5-period SMMA smoothed by 3 periods
    # We use SMMA (Smoothed Moving Average) which is similar to EMA but with different smoothing
    # For simplicity and performance, we'll use EMA as approximation (common in practice)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw_smooth = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values  # Jaw line
    
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth_smooth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values  # Teeth line
    
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips_smooth = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values  # Lips line
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(jaw_smooth[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Alligator Jaw AND 1w close > 1w EMA34 AND volume spike
            if (close[i] > jaw_smooth[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Alligator Jaw AND 1w close < 1w EMA34 AND volume spike
            elif (close[i] < jaw_smooth[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Alligator Jaw (trend weakness)
            if close[i] < jaw_smooth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Alligator Jaw (trend weakness)
            if close[i] > jaw_smooth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals