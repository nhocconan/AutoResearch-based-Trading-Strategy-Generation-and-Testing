#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike confirmation and ATR-based stoploss
# Uses tighter volume confirmation (>2.5x average) to reduce trade frequency
# ATR stoploss (2.0x ATR) manages risk without look-ahead
# Designed for fewer, higher-quality trades with edge in both bull and bear markets

name = "4h_Donchian20_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 20-period Donchian channels (using previous period to avoid look-ahead)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    start_idx = max(20, 14)  # Donchian and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle position management and stoploss
        if position == 1:  # Long position
            # ATR-based stoploss: exit if price drops 2.0x ATR below entry
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss: exit if price rises 2.0x ATR above entry
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.5x 20-period average (stricter)
            vol_confirmed = curr_volume > 2.5 * curr_vol_ma
            
            # Long when price breaks above Donchian high, volume confirmed
            if curr_high > highest_20[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close  # Record entry price for stoploss
            # Short when price breaks below Donchian low, volume confirmed
            elif curr_low < lowest_20[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close  # Record entry price for stoploss
            else:
                signals[i] = 0.0
    
    return signals