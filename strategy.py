#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and volume spike
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) to identify trendless markets.
# Only takes trades when Alligator is "awake" (JAW > TEETH > LIPS for uptrend, reverse for downtrend).
# Confirms with 12h EMA50 trend filter and volume > 2.0x 20-period average.
# Designed for ~25-50 trades/year on 4h timeframe to minimize fee drag while capturing trending moves.
# Works in both bull and bear markets via 12h trend filter - only trades in direction of higher timeframe trend.

name = "4h_WilliamsAlligator_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Alligator on 4h data
    # Jaw = 13-period SMMA, Teeth = 8-period SMMA, Lips = 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator goes to sleep (JAW < TEETH)
            if curr_close < entry_price - 2.0 * curr_atr or curr_jaw < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator goes to sleep (JAW > TEETH)
            if curr_close > entry_price + 2.0 * curr_atr or curr_jaw > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Alligator awake conditions
            jaw_above_teeth = curr_jaw > curr_teeth
            teeth_above_lips = curr_teeth > curr_lips
            jaw_below_teeth = curr_jaw < curr_teeth
            teeth_below_lips = curr_teeth < curr_lips
            
            # Long entry: Alligator awake uptrend (JAW > TEETH > LIPS) + price > 12h EMA50
            if vol_confirm and jaw_above_teeth and teeth_above_lips and curr_close > curr_ema50_12h:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Alligator awake downtrend (JAW < TEETH < LIPS) + price < 12h EMA50
            elif vol_confirm and jaw_below_teeth and teeth_below_lips and curr_close < curr_ema50_12h:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals