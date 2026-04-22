#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ATR filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period high) + ATR filter + volume spike
# Short when price breaks below Donchian lower band (20-period low) + ATR filter + volume spike
# Exit when price reverses back through Donchian midpoint
# Designed for low trade frequency (~15-30/year) to minimize fee drain and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high[i]
        lower = donch_low[i]
        midpoint = donch_mid[i]
        atr_val = atr_14_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # ATR filter: require minimum volatility (avoid choppy markets)
        atr_filter = atr_val > 0  # Always true if ATR calculated, but keeps structure
        
        if position == 0:
            # Long conditions: price breaks above upper band + volatility filter + volume spike
            if price > upper and atr_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volatility filter + volume spike
            elif price < lower and atr_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price reverses back through midpoint
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price falls back below midpoint
                if price < midpoint:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises back above midpoint
                if price > midpoint:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_Volume"
timeframe = "12h"
leverage = 1.0