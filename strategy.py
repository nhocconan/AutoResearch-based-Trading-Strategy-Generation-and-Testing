#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d ADX regime filter + volume confirmation
# In trending markets (1d ADX > 25): trade in direction of Alligator alignment
# In ranging markets (1d ADX <= 25): fade Alligator crossovers (mean reversion)
# Volume confirmation (>1.3x 20-period EMA) filters low-quality signals
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 trades over 4 years.
# Strategy adapts to bull/bear markets via regime filter and uses 12h primary timeframe.

name = "12h_WilliamsAlligator_1dADX_Regime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) with proper min_periods
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    plus_dm = high_1d.diff()
    minus_dm = low_1d.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1d ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # Using EMA as proxy for SMMA with proper min_periods
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_values[i]) or 
            np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            if adx_aligned[i] > 25:
                # Trending market: trade in direction of Alligator alignment
                # Alligator aligned: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
                if lips_values[i] > teeth_values[i] > jaw_values[i]:
                    # Bullish alignment: long
                    if volume_confirm:
                        signals[i] = 0.25
                        position = 1
                elif lips_values[i] < teeth_values[i] < jaw_values[i]:
                    # Bearish alignment: short
                    if volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: fade Alligator crossovers (mean reversion)
                # Bullish crossover: Lips crosses above Teeth
                # Bearish crossover: Lips crosses below Teeth
                if i > 0:
                    lips_prev = lips_values[i-1]
                    teeth_prev = teeth_values[i-1]
                    lips_curr = lips_values[i]
                    teeth_curr = teeth_values[i]
                    
                    # Bullish crossover (Lips crosses above Teeth) -> fade = short
                    if lips_prev <= teeth_prev and lips_curr > teeth_curr and volume_confirm:
                        signals[i] = -0.25
                        position = -1
                    # Bearish crossover (Lips crosses below Teeth) -> fade = long
                    elif lips_prev >= teeth_prev and lips_curr < teeth_curr and volume_confirm:
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Exit long: Alligator misalignment OR ADX weakens (<20) OR volume drops
            if (lips_values[i] <= teeth_values[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator misalignment OR ADX weakens (<20) OR volume drops
            if (lips_values[i] >= teeth_values[i] or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals