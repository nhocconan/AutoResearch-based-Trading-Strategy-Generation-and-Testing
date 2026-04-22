#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-2-3 Reversal with 1w trend filter and volume confirmation
# Uses 1-2-3 reversal pattern: 1st point is swing high/low, 2nd point is retracement, 3rd point breaks beyond 1st point
# Long: 1-2-3 pattern in uptrend (price above 1w EMA50) with volume confirmation
# Short: 1-2-3 pattern in downtrend (price below 1w EMA50) with volume confirmation
# Weekly trend filter provides strong bias, reducing whipsaws
# 12h timeframe targets 12-37 trades/year per symbol (50-150 total over 4 years)
# 1-2-3 pattern is a proven price action setup that works in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Load 1d data for higher timeframe context (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1w EMA(50) for higher timeframe trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate swing points for 1-2-3 pattern (using 1d data for significance)
    # Find swing highs and lows over 3-period window
    window = 3
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(window, n):
        # Swing high: highest high in window
        if high[i] == np.max(high[i-window:i+1]):
            swing_high[i] = high[i]
        # Swing low: lowest low in window
        if low[i] == np.min(low[i-window:i+1]):
            swing_low[i] = low[i]
    
    # Align swing points to 12h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Moderate volume threshold to avoid excessive trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track pattern formation
    bullish_pattern = np.full(n, False)
    bearish_pattern = np.full(n, False)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Detect 1-2-3 bullish pattern: swing low -> retracement -> break above swing high
        if i >= 2:
            # Point 1: swing low
            point1_low = swing_low_aligned[i-2]
            # Point 2: retracement (higher low)
            point2_low = low[i-1]
            # Point 3: break above point 1 (stronger high)
            point3_high = high[i]
            
            if (not np.isnan(point1_low) and 
                point2_low > point1_low and  # Higher low
                point3_high > point1_low):   # Break above point 1
                bullish_pattern[i] = True
        
        # Detect 1-2-3 bearish pattern: swing high -> retracement -> break below swing low
        if i >= 2:
            # Point 1: swing high
            point1_high = swing_high_aligned[i-2]
            # Point 2: retracement (lower high)
            point2_high = high[i-1]
            # Point 3: break below point 1 (stronger low)
            point3_low = low[i]
            
            if (not np.isnan(point1_high) and 
                point2_high < point1_high and  # Lower high
                point3_low < point1_high):     # Break below point 1
                bearish_pattern[i] = True
        
        if position == 0:
            # Long: bullish 1-2-3 pattern + 1w uptrend + volume confirmation
            if (bullish_pattern[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish 1-2-3 pattern + 1w downtrend + volume confirmation
            elif (bearish_pattern[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: pattern completion or trend reversal
            if position == 1:
                # Exit on bearish pattern or trend reversal
                if bearish_pattern[i] or close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish pattern or trend reversal
                if bullish_pattern[i] or close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_123Reversal_1wEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0