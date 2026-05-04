#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d ADX regime + volume confirmation
# Alligator (SMAs with offsets) identifies trend: price > all three lines = uptrend, price < all three = downtrend.
# 1d ADX > 25 confirms strong trend to avoid whipsaws in ranging markets.
# Volume > 1.5x 20-period EMA confirms breakout strength.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_WilliamsAlligator_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = pd.Series(df_1d['high']).sub(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).sub(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).sub(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Williams Alligator on 6h timeframe: SMAs with offsets
    # Jaw (Blue): 13-period SMA, offset 8 bars
    # Teeth (Red): 8-period SMA, offset 5 bars
    # Lips (Green): 5-period SMA, offset 3 bars
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8)
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5)
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3)
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_vals[i]) or 
            np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Alligator alignment: all three lines in order
            # Uptrend: Lips > Teeth > Jaw
            # Downtrend: Lips < Teeth < Jaw
            lips_above_teeth = lips_vals[i] > teeth_vals[i]
            teeth_above_jaw = teeth_vals[i] > jaw_vals[i]
            lips_below_teeth = lips_vals[i] < teeth_vals[i]
            teeth_below_jaw = teeth_vals[i] < jaw_vals[i]
            
            # Strong trend: ADX > 25
            strong_trend = adx_aligned[i] > 25
            
            # Long: Uptrend + strong trend + volume confirmation
            if (lips_above_teeth and teeth_above_jaw and strong_trend and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend + strong trend + volume confirmation
            elif (lips_below_teeth and teeth_below_jaw and strong_trend and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator weakness (Lips < Teeth) OR ADX weakening (<20) OR volume drops
            lips_below_teeth = lips_vals[i] < teeth_vals[i]
            weak_trend = adx_aligned[i] < 20
            low_volume = volume[i] < vol_ema_20[i]
            
            if lips_below_teeth or weak_trend or low_volume:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator weakness (Lips > Teeth) OR ADX weakening (<20) OR volume drops
            lips_above_teeth = lips_vals[i] > teeth_vals[i]
            weak_trend = adx_aligned[i] < 20
            low_volume = volume[i] < vol_ema_20[i]
            
            if lips_above_teeth or weak_trend or low_volume:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals