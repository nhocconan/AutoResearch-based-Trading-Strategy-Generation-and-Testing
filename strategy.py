#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trendless markets when lines intertwine
# Trade only when Alligator is "awake" (lines separated) + price outside lips + 1d EMA34 trend alignment
# Volume spike >1.8x confirms institutional participation; discrete sizing 0.25 minimizes fee churn
# Works in bull/bear: avoids choppy markets, catches strong trends with volume validation

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs of median price (hl2) with specific offsets
    hl2 = (high + low) / 2.0
    jaw = pd.Series(hl2).rolling(window=13, min_periods=13).mean().shift(8).values  # Jaw: 13-period, shifted 8
    teeth = pd.Series(hl2).rolling(window=8, min_periods=8).mean().shift(5).values   # Teeth: 8-period, shifted 5
    lips = pd.Series(hl2).rolling(window=5, min_periods=5).mean().shift(3).values    # Lips: 5-period, shifted 3
    
    # Alligator awake condition: lips outside jaw-teeth or teeth outside jaw-lips (trending)
    # Simplified: lips > max(jaw, teeth) OR lips < min(jaw, teeth) = strong separation
    lips_above = lips > np.maximum(jaw, teeth)
    lips_below = lips < np.minimum(jaw, teeth)
    alligator_awake = lips_above | lips_below
    
    # Direction: price relative to lips (trend direction)
    price_above_lips = close > lips
    price_below_lips = close < lips
    
    # 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.8 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_alligator_awake = alligator_awake[i]
        curr_price_above_lips = price_above_lips[i]
        curr_price_below_lips = price_below_lips[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade when Alligator is awake (trending) + volume confirmation
            if curr_alligator_awake and curr_volume_confirm:
                # Bullish entry: price above lips + above 1d EMA34
                if curr_price_above_lips and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price below lips + below 1d EMA34
                elif curr_price_below_lips and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price crosses below lips (trend weakening)
            if curr_close < curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above lips (trend weakening)
            if curr_close > curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals