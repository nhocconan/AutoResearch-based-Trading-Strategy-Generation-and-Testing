#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 12h identifies trend direction and alignment.
Alligator lines aligned (Lips > Teeth > Jaw for uptrend, reverse for downtrend) with volume confirmation
and 1d EMA50 trend filter captures strong momentum moves. Works in bull markets via long alignments
and in bear markets via short alignments. ATR-based stoploss manages risk. Targets 50-150 total trades
over 4 years on 12h timeframe to avoid fee drag.
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
    
    # Get 12h data for Williams Alligator and 1d data for EMA50 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: Jaw=13*8, Teeth=8*5, Lips=5*3 SMAs of median price
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_12h).rolling(window=13*8, min_periods=13*8).mean().values
    teeth = pd.Series(median_12h).rolling(window=8*5, min_periods=8*5).mean().values
    lips = pd.Series(median_12h).rolling(window=5*3, min_periods=5*3).mean().values
    
    # Align Alligator lines to 12h timeframe (already 12h, so direct use)
    jaw_12h = jaw
    teeth_12h = teeth
    lips_12h = lips
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    start_idx = max(13*8, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_12h[i]
        teeth_val = teeth_12h[i]
        lips_val = lips_12h[i]
        ema50_1d = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Long alignment: Lips > Teeth > Jaw with volume confirmation and 1d EMA50 uptrend
            long_align = (lips_val > teeth_val) and (teeth_val > jaw_val) and volume_confirm and (curr_close > ema50_1d)
            # Short alignment: Lips < Teeth < Jaw with volume confirmation and 1d EMA50 downtrend
            short_align = (lips_val < teeth_val) and (teeth_val < jaw_val) and volume_confirm and (curr_close < ema50_1d)
            
            if long_align:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_align:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0