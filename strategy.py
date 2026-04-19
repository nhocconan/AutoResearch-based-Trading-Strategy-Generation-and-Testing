#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123 Reversal pattern with 1d volume confirmation and 1w trend filter
# - 123 Reversal: Long when price makes higher low (HL) and breaks above prior high (PH)
#                 Short when price makes lower high (LH) and breaks below prior low (PL)
# - Uses 14-bar swing detection to identify points 1, 2, 3
# - Volume filter: 4h volume > 1.5x 20-period 1d average volume (scaled to 4h)
# - Trend filter: Only take longs in uptrend (price > 1w EMA50), shorts in downtrend (price < 1w EMA50)
# - Designed to catch trend reversals with confirmation, works in both bull and bear markets
# - Target: 25-40 trades/year to avoid excessive fee drag

name = "4h_123Reversal_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Swing detection for 123 pattern (14-bar lookback)
    # Point 2: recent swing high/low
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    # Point 1: swing before point 2 (28-bar lookback for context)
    highest_high_28 = pd.Series(high).rolling(window=28, min_periods=28).max()
    lowest_low_28 = pd.Series(low).rolling(window=28, min_periods=28).min()
    
    # Point 3: current price testing the swing
    
    # Calculate swing points
    # For bullish 123: Point 1 (low), Point 2 (higher low), Point 3 (break above Point 2 high)
    # For bearish 123: Point 1 (high), Point 2 (lower high), Point 3 (break below Point 2 low)
    
    # Simplified approach: detect higher lows and lower highs
    # Higher Low: current low > prior low AND prior low < low before that
    # Lower High: current high < prior high AND prior high > high before that
    
    hl_condition = (low > np.roll(low, 1)) & (np.roll(low, 1) < np.roll(low, 2))
    lh_condition = (high < np.roll(high, 1)) & (np.roll(high, 1) > np.roll(high, 2))
    
    # Breakout conditions
    # Bullish: price breaks above the high of the higher low setup
    # Bearish: price breaks below the low of the lower high setup
    
    # Track recent swing highs/lows for breakout levels
    recent_high = np.maximum.accumulate(high)
    recent_low = np.minimum.accumulate(low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for swing detection
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Look for bullish 123: Higher Low formation + breakout + uptrend + volume
            if (hl_condition[i] and 
                close[i] > recent_high[i-1] and  # Break above prior high
                close[i] > ema_50_1w_aligned[i] and  # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for bearish 123: Lower High formation + breakdown + downtrend + volume
            elif (lh_condition[i] and 
                  close[i] < recent_low[i-1] and  # Break below prior low
                  close[i] < ema_50_1w_aligned[i] and  # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown or trend reversal
            if close[i] < recent_low[i-1] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout or trend reversal
            if close[i] > recent_high[i-1] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals