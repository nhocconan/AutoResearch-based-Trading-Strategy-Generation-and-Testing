#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R (14) extreme reversal with 1w EMA(34) trend filter and volume confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1w EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Williams %R: Long when %R crosses above -80 from below (oversold reversal), short when %R crosses below -20 from above (overbought reversal).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm momentum.
- Exit: Opposite Williams %R extreme (%R < -80 for longs, %R > -20 for shorts) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why it should work: Williams %R captures short-term exhaustion in both bull and bear markets, while 1w EMA ensures alignment with the major trend. Volume confirmation avoids fakeouts. Effective in ranging and trending conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1w data for EMA(34) trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1w
    vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)  # Need enough 1w bars for EMA34 and volume MA, and 6h for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish reversal: Williams %R crosses above -80 from below (oversold)
                if prev_williams_r <= -80 and curr_williams_r > -80 and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above (overbought)
                elif prev_williams_r >= -20 and curr_williams_r < -20 and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns below -80 (loss of momentum) OR loss of volume confirmation
            if curr_williams_r < -80 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns above -20 (loss of momentum) OR loss of volume confirmation
            if curr_williams_r > -20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_1wEMA34Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0