#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation (2.0x)
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-period average
# Exit when Alligator alignment reverses OR price crosses 1d EMA50
# Williams Alligator provides trend identification with built-in smoothing effective in both bull and bear markets
# 1d EMA50 provides higher timeframe trend filter
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1d

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_2.0x"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using close prices)
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods
    # Lips: 5-period SMMA smoothed by 3 periods
    if len(close) >= 13:
        jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
        jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
        
        teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
        teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
        
        lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
        lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    else:
        jaw = teeth = lips = np.full(n, np.nan)
    
    # Volume confirmation on 12h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: jaw < teeth < lips
            bullish_alignment = jaw[i] < teeth[i] < lips[i]
            # Bearish Alligator alignment: jaw > teeth > lips
            bearish_alignment = jaw[i] > teeth[i] > lips[i]
            
            # Long: bullish alignment AND price > EMA50 AND volume spike
            if (bullish_alignment and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < EMA50 AND volume spike
            elif (bearish_alignment and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment turns bearish OR price < EMA50 (trend weakening)
            bearish_alignment = jaw[i] > teeth[i] > lips[i]
            if bearish_alignment or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment turns bullish OR price > EMA50 (trend weakening)
            bullish_alignment = jaw[i] < teeth[i] < lips[i]
            if bullish_alignment or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals