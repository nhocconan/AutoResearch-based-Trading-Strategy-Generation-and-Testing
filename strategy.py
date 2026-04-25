#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets on 12h.
In trending markets (Alligator aligned), breakouts in direction of 1d EMA50 with volume spike
capture strong momentum moves. Works in bull markets via longs above lips, bear via shorts below lips.
ATR-based trailing stop manages risk. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
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
    
    # Williams Alligator on 12h: jaw=13*8, teeth=8*5, lips=5*3 (smoothed with SMA)
    jaw_period = 13 * 8   # 104
    teeth_period = 8 * 5  # 40
    lips_period = 5 * 3   # 15
    
    # Calculate smoothed median price (typical price) for Alligator
    typical_price = (high + low + close) / 3.0
    
    # Jaw (blue line): 104-period SMA of typical price, shifted 8 bars forward
    jaw = np.full(n, np.nan)
    for i in range(jaw_period - 1, n):
        jaw[i] = np.mean(typical_price[i - jaw_period + 1:i + 1])
    # Shift jaw by 8 bars forward (to align with Bill Williams' method)
    jaw_shifted = np.full(n, np.nan)
    jaw_shifted[8:] = jaw[:-8] if n > 8 else jaw
    
    # Teeth (red line): 40-period SMA of typical price, shifted 5 bars forward
    teeth = np.full(n, np.nan)
    for i in range(teeth_period - 1, n):
        teeth[i] = np.mean(typical_price[i - teeth_period + 1:i + 1])
    # Shift teeth by 5 bars forward
    teeth_shifted = np.full(n, np.nan)
    teeth_shifted[5:] = teeth[:-5] if n > 5 else teeth
    
    # Lips (green line): 15-period SMA of typical price, shifted 3 bars forward
    lips = np.full(n, np.nan)
    for i in range(lips_period - 1, n):
        lips[i] = np.mean(typical_price[i - lips_period + 1:i + 1])
    # Shift lips by 3 bars forward
    lips_shifted = np.full(n, np.nan)
    lips_shifted[3:] = lips[:-3] if n > 3 else lips
    
    # Calculate 20-period volume MA for volume confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (12h)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Alligator, EMA50_1d, volume MA, ATR to propagate
    start_idx = max(jaw_period + 8, teeth_period + 5, lips_period + 3, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Alligator alignment: jaws < teeth < lips = uptrend, jaws > teeth > lips = downtrend
        alligator_uptrend = (jaw_val < teeth_val) and (teeth_val < lips_val)
        alligator_downtrend = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: price above lips, Alligator uptrend, volume confirmation, 1d EMA50 uptrend
            long_entry = (curr_close > lips_val) and alligator_uptrend and volume_confirm and (curr_close > ema50_1d)
            # Short entry: price below lips, Alligator downtrend, volume confirmation, 1d EMA50 downtrend
            short_entry = (curr_close < lips_val) and alligator_downtrend and volume_confirm and (curr_close < ema50_1d)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop OR Alligator loses uptrend alignment
            if (curr_close < atr_stop) or not alligator_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop OR Alligator loses downtrend alignment
            if (curr_close > atr_stop) or not alligator_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0