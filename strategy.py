#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) + 1d EMA50 trend filter + volume confirmation
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1d EMA50 AND volume > 2.0x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1d EMA50 AND volume > 2.0x 20-period average
# Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw) OR close crosses 1d EMA50
# Uses 12h primary timeframe with 1d HTF for trend filter to capture sustained moves with low frequency
# Volume confirmation ensures institutional participation; discrete sizing (0.25) limits fee drag
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag in ranging markets
# Alligator identifies trend absence (all lines intertwined) vs presence (diverged); EMA50 filters higher-timeframe direction

name = "12h_Williams_Alligator_1dEMA50_Trend_Volume"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components on 12h data
    # Jaw: Blue line - 13-period SMMA smoothed 8 periods ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    # Teeth: Red line - 8-period SMMA smoothed 5 periods ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    # Lips: Green line - 5-period SMMA smoothed 3 periods ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
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
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) 
            #              AND close > 1d EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment)
            #               AND close < 1d EMA50 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips <= Teeth OR Teeth <= Jaw) 
            #          OR close < 1d EMA50 (trend flip)
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips >= Teeth OR Teeth >= Jaw) 
            #          OR close > 1d EMA50 (trend flip)
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals