#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 123 Reversal + 1d Trend + Volume Spike
# Hypothesis: 123 Reversal pattern captures institutional reversal points
# in both bull and bear markets. Combined with 1d EMA trend filter and
# volume confirmation, it filters false signals. Target: 20-40 trades/year.

name = "4h_123_reversal_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and 123 pattern
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 123 Reversal pattern detection
    # Point 1: swing high/low
    # Point 2: retracement
    # Point 3: failure to exceed point 1
    lookback = 10
    
    # Find swing points
    highs = pd.Series(high).rolling(window=lookback, center=False).max().values
    lows = pd.Series(low).rolling(window=lookback, center=False).min().values
    
    # Point 1: recent swing high/low
    pt1_high = np.roll(highs, 1)
    pt1_low = np.roll(lows, 1)
    
    # Point 2: pullback from point 1
    pt2_high = pt1_high - 0.382 * (pt1_high - np.roll(lows, 2))  # 38.2% retracement
    pt2_low = pt1_low + 0.382 * (np.roll(highs, 2) - pt1_low)
    
    # Point 3: test of point 2 area
    pt3_high = pt2_high + 0.5 * (pt1_high - pt2_high)  # 50% of retracement
    pt3_low = pt2_low - 0.5 * (pt1_low - pt2_low)
    
    # Align to 4h timeframe
    pt1_high_aligned = align_htf_to_ltf(prices, df_1d, pt1_high)
    pt1_low_aligned = align_htf_to_ltf(prices, df_1d, pt1_low)
    pt2_high_aligned = align_htf_to_ltf(prices, df_1d, pt2_high)
    pt2_low_aligned = align_htf_to_ltf(prices, df_1d, pt2_low)
    pt3_high_aligned = align_htf_to_ltf(prices, df_1d, pt3_high)
    pt3_low_aligned = align_htf_to_ltf(prices, df_1d, pt3_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pt1_high_aligned[i]) or np.isnan(pt1_low_aligned[i]) or
            np.isnan(pt2_high_aligned[i]) or np.isnan(pt2_low_aligned[i]) or
            np.isnan(pt3_high_aligned[i]) or np.isnan(pt3_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below point 3 or trend turns bearish
            if low[i] < pt3_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above point 3 or trend turns bullish
            if high[i] > pt3_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish 123 reversal: point 3 holds above point 2, uptrend
                if (low[i] > pt2_low_aligned[i] and 
                    high[i] < pt1_high_aligned[i] and
                    close[i] > pt2_high_aligned[i] and
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Bearish 123 reversal: point 3 holds below point 2, downtrend
                elif (high[i] < pt2_high_aligned[i] and 
                      low[i] > pt1_low_aligned[i] and
                      close[i] < pt2_low_aligned[i] and
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals