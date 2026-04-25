#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw=TEETH=LIPS) identifies trend absence/presence; when all three lines are aligned (trending) and price is outside the Alligator's mouth, we have a strong trend. Combined with 1d EMA50 trend filter and volume spike, this captures sustained moves in both bull and bear markets. The Alligator's sleep/awake cycle acts as a natural regime filter, reducing whipsaws in ranging markets. Targets 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
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
    
    # Get 1d data for EMA50 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (typical price)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    # We'll use EMA as approximation for SMMA (common practice)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (blue)
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values    # Teeth (red)
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values     # Lips (green)
    
    # ATR(21) for stoploss and volatility
    if len(close) >= 21:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=21, min_periods=21).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator lines, EMA50_1d, ATR, and volume MA
    start_idx = max(13, 50, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50_1d = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator alignment: check if all three lines are ordered (trending)
        # Jaw > Teeth > Lips = uptrend alignment
        # Jaw < Teeth < Lips = downtrend alignment
        jaw_above_teeth = jaw_val > teeth_val
        teeth_above_lips = teeth_val > lips_val
        jaw_below_teeth = jaw_val < teeth_val
        teeth_below_lips = teeth_val < lips_val
        
        # Uptrend: Jaw > Teeth > Lips
        uptrend_aligned = jaw_above_teeth and teeth_above_lips
        # Downtrend: Jaw < Teeth < Lips
        downtrend_aligned = jaw_below_teeth and teeth_below_lips
        
        # Price outside Alligator's mouth (strong trend signal)
        # Mouth is between Jaw and Lips
        price_above_mouth = curr_close > max(jaw_val, lips_val)
        price_below_mouth = curr_close < min(jaw_val, lips_val)
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: uptrend aligned AND price above mouth AND above 1d EMA50 AND volume spike
            long_condition = uptrend_aligned and price_above_mouth and (curr_close > ema50_1d) and volume_spike
            # Short: downtrend aligned AND price below mouth AND below 1d EMA50 AND volume spike
            short_condition = downtrend_aligned and price_below_mouth and (curr_close < ema50_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or Alligator starts sleeping (jaw < teeth)
            if curr_close <= entry_price - 2.5 * atr_val or jaw_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or Alligator starts sleeping (jaw > teeth)
            if curr_close >= entry_price + 2.5 * atr_val or jaw_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0