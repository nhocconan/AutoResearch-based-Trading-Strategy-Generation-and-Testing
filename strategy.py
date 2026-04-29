#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) + 1d EMA50 Trend Filter + Volume Spike
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x avg
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x avg
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter.
# Timeframe: 12h (primary), HTF: 1d for EMA50 trend.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (using close)
    # Jaw: SMA(13, 8) - 13 period, 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5) - 8 period, 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3) - 5 period, 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13+8, 8+5, 5+3, 20)  # warmup for EMA50, Alligator shifts, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            # 2. Price falls below 1d EMA50 (trend change)
            if (curr_lips <= curr_teeth or
                curr_teeth <= curr_jaw or
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            # 2. Price rises above 1d EMA50 (trend change)
            if (curr_lips >= curr_teeth or
                curr_teeth >= curr_jaw or
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume confirm
            if (curr_lips > curr_teeth and
                curr_teeth > curr_jaw and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume confirm
            elif (curr_lips < curr_teeth and
                  curr_teeth < curr_jaw and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals