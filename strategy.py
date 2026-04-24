#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Alligator Jaw (13-period smoothed median), Teeth (8-period smoothed median), Lips (5-period smoothed median).
- Volume: Current 4h volume > 1.5 * 20-period volume MA to confirm momentum.
- Entry: Long when Alligator is bullish (Lips > Teeth > Jaw) AND price > 1d EMA34 AND volume spike.
         Short when Alligator is bearish (Lips < Teeth < Jaw) AND price < 1d EMA34 AND volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams Alligator identifies trending vs ranging markets; works in both bull and bear by filtering with 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price (typical price) for Alligator
    median_price = (high + low + close) / 3.0
    
    # Alligator Jaw: 13-period SMMA of median price (shifted 8 bars ahead)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Alligator Teeth: 8-period SMMA of median price (shifted 5 bars ahead)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Alligator Lips: 5-period SMMA of median price (shifted 3 bars ahead)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 34, 20)  # Need enough bars for Alligator and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        ema_val = ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Lips > Teeth > Jaw (Alligator bullish) AND price > 1d EMA34
                if lips_val > teeth_val > jaw_val and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw (Alligator bearish) AND price < 1d EMA34
                elif lips_val < teeth_val < jaw_val and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR loss of volume confirmation
            if not (lips_val > teeth_val > jaw_val) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR loss of volume confirmation
            if not (lips_val < teeth_val < jaw_val) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0