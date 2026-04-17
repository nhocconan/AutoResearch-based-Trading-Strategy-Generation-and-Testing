#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Williams Alligator (Jaw/Teeth/Lips) for trend direction,
combined with 1d volume spike confirmation and ATR-based trailing stop.
- Long when Alligator lines are bullish (Lips > Teeth > Jaw) + volume > 2.0x 20-period 1d volume MA
- Short when Alligator lines are bearish (Lips < Teeth < Jaw) + volume > 2.0x 20-period 1d volume MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.5x ATR) to lock in profits
- Designed for low trade frequency (<100 trades over 4 years) to avoid fee drag
- Works in bull markets (catching trends) and bear markets (shorting breakdowns)
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
    
    # Get 4h data for Alligator (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams Alligator: Smoothed Moving Average (SMA) with specific periods
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward 8 bars
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward 5 bars
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward 3 bars
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Get 1d data for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period volume moving average on 1d
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_vals)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Bullish Alligator: Lips > Teeth > Jaw
        bullish = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish Alligator: Lips < Teeth < Jaw
        bearish = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Look for Alligator alignment with volume confirmation
            # Long: Bullish Alligator + volume spike
            if bullish and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: Bearish Alligator + volume spike
            elif bearish and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 2.0 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 2.0 * atr_val)
    
    return signals

name = "4h_WilliamsAlligator_1dVolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0