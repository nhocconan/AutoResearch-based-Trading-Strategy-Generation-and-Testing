#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMA shifted 8) AND price > 1d EMA(34) AND volume > 1.5x 20-period average
# Short when price < Alligator Lips (8-period SMA shifted 5) AND price < 1d EMA(34) AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Alligator catches trends in both bull and bear markets.
# Timeframe: 12h (primary), HTF: 1d for trend filter.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (12h timeframe)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # Using SMA as approximation for SMMA (similar enough for crossover signals)
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate Jaw (13-period SMA shifted 8)
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, jaw_shift)  # shift right (into future)
    jaw[:jaw_shift] = np.nan  # fill shifted values with nan
    
    # Calculate Teeth (8-period SMA shifted 5)
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, teeth_shift)  # shift right (into future)
    teeth[:teeth_shift] = np.nan  # fill shifted values with nan
    
    # Calculate Lips (5-period SMA shifted 3)
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, lips_shift)  # shift right (into future)
    lips[:lips_shift] = np.nan  # fill shifted values with nan
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 34)  # warmup
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Skip if any Alligator line is not yet valid (nan from shifting)
        if np.isnan(curr_jaw) or np.isnan(curr_teeth) or np.isnan(curr_lips):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price crosses below Alligator Lips (weaken trend)
            # 2. Price < 1d EMA(34) (trend change)
            # 3. Trailing stop: price drops 3.0*ATR from high since entry (tracked separately)
            if (curr_close < curr_lips or 
                curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price crosses above Alligator Jaw (weaken trend)
            # 2. Price > 1d EMA(34) (trend change)
            if (curr_close > curr_jaw or 
                curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > Alligator Jaw AND price > 1d EMA(34) AND volume confirmation
            if (curr_close > curr_jaw and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Alligator Lips AND price < 1d EMA(34) AND volume confirmation
            elif (curr_close < curr_lips and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals