#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator alignment with 1d EMA34 trend filter and volume confirmation.
Long when Alligator lines are bullish (Lips > Teeth > Jaw) AND price > 1d EMA34 AND volume > 1.3x 20-period average.
Short when Alligator lines are bearish (Lips < Teeth < Jaw) AND price < 1d EMA34 AND volume > 1.3x 20-period average.
Exit when Alligator alignment reverses (Lips crosses Teeth) or price crosses 1d EMA34.
Uses 1d HTF for EMA trend strength to avoid whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
Williams Alligator identifies trend initiation and continuation; EMA34 filter ensures alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: Blue line - 13-period SMMA smoothed, shifted 8 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift ahead
    
    # Teeth: Red line - 8-period SMMA smoothed, shifted 5 bars ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift ahead
    
    # Lips: Green line - 5-period SMMA smoothed, shifted 3 bars ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift ahead
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 20, 34)  # jaw(13+8), teeth(8+5), lips(5+3), vol(20), ema(34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish alignment AND price > 1d EMA34 AND volume spike
            if bullish_alignment and price > ema_trend and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < 1d EMA34 AND volume spike
            elif bearish_alignment and price < ema_trend and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reversal OR price crosses 1d EMA34
            lips_teeth_cross = (position == 1 and lips[i] < teeth[i]) or (position == -1 and lips[i] > teeth[i])
            price_ema_cross = (position == 1 and price < ema_trend) or (position == -1 and price > ema_trend)
            
            if lips_teeth_cross or price_ema_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_Alignment_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0