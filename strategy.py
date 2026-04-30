#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trend absence (all lines intertwined).
# Enters when price breaks above/below Alligator lips in direction of 1d EMA50 with volume spike (>2.0x 20-bar avg).
# Exits when price re-enters Alligator mouth (between Jaw and Teeth) or ATR trailing stop (2.0*ATR).
# Discrete position sizing at ±0.25 to minimize fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid overtrading on 12h.
# Works in bull markets via trend continuation and in bear markets via volatility expansion capture.
# Session filter (08:00-20:00 UTC) avoids low-liquidity periods.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (Jaw=13, Teeth=8, Lips=5)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(5).values
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(3).values
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(jaw_period, teeth_period, lips_period) + 8  # warmup for Alligator and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price above lips, lips above teeth, teeth above jaw (aligned Alligator), above 1d EMA50, volume spike
            if (curr_close > curr_lips and 
                curr_lips > curr_teeth and 
                curr_teeth > curr_jaw and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price below lips, lips below teeth, teeth below jaw (aligned Alligator), below 1d EMA50, volume spike
            elif (curr_close < curr_lips and 
                  curr_lips < curr_teeth and 
                  curr_teeth < curr_jaw and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price re-enters Alligator mouth (between Jaw and Teeth) OR ATR trailing stop
            if (curr_close < curr_teeth) or (curr_close < highest_since_entry - (2.0 * curr_atr)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price re-enters Alligator mouth (between Jaw and Teeth) OR ATR trailing stop
            if (curr_close > curr_teeth) or (curr_close > lowest_since_entry + (2.0 * curr_atr)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals