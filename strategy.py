#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend Filter + Volume Spike Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets. 
When Lips cross above Teeth (bullish alignment) with price above 1d EMA50 and volume spike (>2.0x 20-bar vol MA) = long.
When Lips cross below Teeth (bearish alignment) with price below 1d EMA50 and volume spike = short.
Uses proper MTF alignment for 1d EMA50 and discrete sizing (0.25) to limit fee drag.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Works in bull markets via upside alignment and in bear markets via downside alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 51:  # Need 50 for EMA + 1 for safety
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h data: SMAs with specific periods
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # Using EMA as proxy for SMMA (similar smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8), EMA50, and volume MA
    start_idx = max(21, 51, 20)  # 21 for Alligator (13+8), 51 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaws_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_align = (lips_val > teeth_val) and (teeth_val > jaws_val)
        bearish_align = (lips_val < teeth_val) and (teeth_val < jaws_val)
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: bullish alignment + price above 1d EMA50 + volume confirmation
            long_signal = bullish_align and price_above_ema and volume_confirm
            # Short: bearish alignment + price below 1d EMA50 + volume confirmation
            short_signal = bearish_align and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR price crosses below 1d EMA50
            if bearish_align or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR price crosses above 1d EMA50
            if bullish_align or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0