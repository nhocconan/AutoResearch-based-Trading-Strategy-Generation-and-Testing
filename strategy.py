#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend Filter + Volume Spike Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence/presence on 6h. Combined with 1d EMA50 for higher timeframe trend filter and volume spike (>2.0x 20-bar vol MA) to capture strong momentum moves. Works in bull markets via long when lips>teeth>jaw and price above EMA50; in bear markets via short when lips<teeth<jaw and price below EMA50. Targeting 12-30 trades per year to avoid fee drag while maintaining edge in both regimes.
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
    if len(df_1d) < 50:  # Need 50 for EMA
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: SMA of median price (HL/2)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_price = (high + low) / 2
    
    # Jaw (blue line)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red line)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green line)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator, EMA50, and volume MA
    start_idx = max(16, 50, 20)  # 16 for lips (5+3 shift), 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Alligator alignment: bullish (lips>teeth>jaw) or bearish (lips<teeth<jaw)
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: bullish Alligator + price above 1d EMA50 + volume confirmation
            long_signal = alligator_bullish and price_above_ema and volume_confirm
            # Short: bearish Alligator + price below 1d EMA50 + volume confirmation
            short_signal = alligator_bearish and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR price crosses below 1d EMA50
            if not alligator_bullish or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR price crosses above 1d EMA50
            if not alligator_bearish or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0