#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1w EMA(50) trend filter and volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1w EMA(50) for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Williams Alligator: Jaw (SMA 13, 8-shift), Teeth (SMA 8, 5-shift), Lips (SMA 5, 3-shift).
- Volume: Current 12h volume > 2.0 * 20-period volume MA to avoid false signals.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > EMA50 (1w) AND volume spike.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < EMA50 (1w) AND volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h
    # Jaw: SMA(13, 8) - median price smoothed with 13-period, shifted 8 bars
    # Teeth: SMA(8, 5) - median price smoothed with 8-period, shifted 5 bars
    # Lips: SMA(5, 3) - median price smoothed with 5-period, shifted 3 bars
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1w data for EMA(50) trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 1w
    vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13)  # Need enough 1w bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish alignment: Lips > Teeth > Jaw AND price > EMA50 (1w)
                if lips[i] > teeth[i] > jaw[i] and curr_close > ema_50_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaw AND price < EMA50 (1w)
                elif lips[i] < teeth[i] < jaw[i] and curr_close < ema_50_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish alignment OR loss of volume confirmation
            if lips[i] < teeth[i] < jaw[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR loss of volume confirmation
            if lips[i] > teeth[i] > jaw[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0