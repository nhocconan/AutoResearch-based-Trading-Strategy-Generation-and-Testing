#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + ATR Stop
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends via SMAs
# Jaw (13-period SMA shifted 8 bars), Teeth (8-period SMA shifted 5 bars), Lips (5-period SMA shifted 3 bars)
# Bullish: Lips > Teeth > Jaw; Bearish: Jaw > Teeth > Lips
# Volume Spike: 1d volume > 2x 20-period average
# ATR Stop: Exit when price moves against position by 2x ATR
# Works in both bull/bear: Alligator defines trend, volume confirms strength
# 12h timeframe targets 12-37 trades/year (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Alligator and Volume Spike
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator components on 1d
    close_1d = df_1d['close'].values
    # Jaw: 13-period SMA shifted 8 bars
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = np.concatenate([np.full(8, np.nan), jaw_raw.values[:-8]]) if len(jaw_raw) > 8 else np.full_like(close_1d, np.nan)
    # Teeth: 8-period SMA shifted 5 bars
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = np.concatenate([np.full(5, np.nan), teeth_raw.values[:-5]]) if len(teeth_raw) > 5 else np.full_like(close_1d, np.nan)
    # Lips: 5-period SMA shifted 3 bars
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = np.concatenate([np.full(3, np.nan), lips_raw.values[:-3]]) if len(lips_raw) > 3 else np.full_like(close_1d, np.nan)
    
    # 1d Volume Spike: volume > 2x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (2.0 * vol_ma)
    
    # Align Williams Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0=flat, 1=long, -1=short
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Williams Alligator signals
        bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Enter long: bullish Alligator + volume spike
            if bullish and vol_spike_aligned[i] > 0.5:
                position = 1
                signals[i] = position_size
            # Enter short: bearish Alligator + volume spike
            elif bearish and vol_spike_aligned[i] > 0.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish Alligator OR stoploss hit
            if bearish or (price <= entry_price - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish Alligator OR stoploss hit
            if bullish or (price >= entry_price + 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        
        # Track entry price for stoploss
        if position != 0 and signals[i] != 0.0 and (i == start or signals[i-1] == 0.0):
            entry_price = price
    
    return signals

name = "12h_WilliamsAlligator_1dVolSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0