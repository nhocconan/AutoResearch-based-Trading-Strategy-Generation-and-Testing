#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter with volume spike confirmation
# Uses 6h timeframe for Alligator (jaw/teeth/lips) to identify trend direction and entry timing
# 1d EMA34 provides higher-timeframe trend filter to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Williams Alligator is effective in both trending and ranging markets:
#   - In trends: lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
#   - In ranges: lines intertwine → no trades taken (avoids whipsaws)
# Combined with 1d trend filter and volume confirmation for high-probability entries

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h timeframe (smoothed medians)
    # Jaw: 13-period SMMA, 8-period offset
    # Teeth: 8-period SMMA, 5-period offset  
    # Lips: 5-period SMMA, 3-period offset
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high + low, 13)  # Typical price for Alligator
    jaw = np.roll(jaw, 8)       # 8-period offset
    
    teeth = smma(high + low, 8) 
    teeth = np.roll(teeth, 5)   # 5-period offset
    
    lips = smma(high + low, 5)
    lips = np.roll(lips, 3)     # 3-period offset
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: check if lines are properly separated (trending condition)
        # Bullish: lips > teeth > jaw
        # Bearish: lips < teeth < jaw
        bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment + price > 1d EMA34 + volume confirm
            if bullish_align and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price < 1d EMA34 + volume confirm
            elif bearish_align and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines reverse (lips < teeth) or price crosses below teeth
            if lips[i] < teeth[i] or close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines reverse (lips > teeth) or price crosses above teeth
            if lips[i] > teeth[i] or close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals