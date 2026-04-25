#!/usr/bin/env python3
"""
12h Williams Alligator + 1w EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator identifies trend presence and direction via smoothed medians (Jaw/Teeth/Lips). 
1w EMA50 filters for primary trend alignment to avoid counter-trend trades. Volume spike confirms institutional participation.
Works in bull via buying when Lips > Teeth > Jaw (bullish alignment) and price > 1w EMA50. 
Works in bear via selling when Lips < Teeth < Jaw (bearish alignment) and price < 1w EMA50.
Uses discrete position sizing (0.25) to control drawdown and minimize fee churn.
Target: 12-37 trades/year on 12h (50-150 total over 4 years).
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
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Williams Alligator: Smoothed medians (Jaw=13, Teeth=8, Lips=5)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw (13-period SMMA of median)
    jaw = np.full(n, np.nan)
    if n >= 13:
        jaw[12] = np.mean(median_price[:13])
        for i in range(13, n):
            jaw[i] = (jaw[i-1] * 12 + median_price[i]) / 13
    
    # Teeth (8-period SMMA of median)
    teeth = np.full(n, np.nan)
    if n >= 8:
        teeth[7] = np.mean(median_price[:8])
        for i in range(8, n):
            teeth[i] = (teeth[i-1] * 7 + median_price[i]) / 8
    
    # Lips (5-period SMMA of median)
    lips = np.full(n, np.nan)
    if n >= 5:
        lips[4] = np.mean(median_price[:5])
        for i in range(5, n):
            lips[i] = (lips[i-1] * 4 + median_price[i]) / 5
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator (13) and EMA50 to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_1w_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Alligator signals: Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Bullish Alligator AND price > 1w EMA50 AND volume spike
            long_condition = bullish_alligator and curr_close > ema_50 and volume_spike
            # Short: Bearish Alligator AND price < 1w EMA50 AND volume spike
            short_condition = bearish_alligator and curr_close < ema_50 and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Alligator turns bearish or price < 1w EMA50
            if curr_close <= entry_price - 2.0 * atr_val or not bullish_alligator or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Alligator turns bullish or price > 1w EMA50
            if curr_close >= entry_price + 2.0 * atr_val or not bearish_alligator or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0