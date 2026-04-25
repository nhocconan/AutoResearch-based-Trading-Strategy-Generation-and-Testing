#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets on 6h.
In trending markets (Alligator aligned), enter on pullback to teeth (EMA8) in direction of 1d EMA50 trend.
Volume confirmation filters weak signals. Works in bull/bear via trend-following pullbacks.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: jaw=EMA(13,8), teeth=EMA(8,5), lips=EMA(5,3)
    # Using Smoothed Moving Average (SMMA) approximation via EMA with adjusted span
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    if len(close) >= 13:
        # Jaw: EMA(13,8) -> EMA with span=13, then smoothed 8 periods
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
        jaw = ema13.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    if len(close) >= 8:
        # Teeth: EMA(8,5) -> EMA with span=8, then smoothed 5 periods
        ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
        teeth = ema8.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    if len(close) >= 5:
        # Lips: EMA(5,3) -> EMA with span=5, then smoothed 3 periods
        ema5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
        lips = ema5.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator, 1d EMA50, and volume MA to propagate
    start_idx = max(50, 20, 13)  # 1d EMA50 needs 50, volume MA needs 20, Alligator jaw needs 13
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema50_1d = ema_50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        volume_confirm = curr_volume > 1.3 * vol_ma
        
        # Alligator alignment: check if trending (jaws, teeth, lips separated and aligned)
        # Bullish alignment: lips > teeth > jaw (alligator mouth opening up)
        # Bearish alignment: jaw > teeth > lips (alligator mouth opening down)
        bullish_align = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_align = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        if position == 0:
            # Long: bullish alignment AND price above teeth (pullback entry) AND 1d EMA50 uptrend AND volume confirmation
            long_condition = bullish_align and (curr_close > teeth_val) and (curr_close > ema50_1d) and volume_confirm
            # Short: bearish alignment AND price below teeth (pullback entry) AND 1d EMA50 downtrend AND volume confirmation
            short_condition = bearish_align and (curr_close < teeth_val) and (curr_close < ema50_1d) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lips (Alligator wake-up signal) or trend reversal
            if curr_close < lips_val or not bullish_align:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above lips (Alligator wake-up signal) or trend reversal
            if curr_close > lips_val or not bearish_align:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0