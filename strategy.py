#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d RSI trend filter and volume confirmation
# Long when price breaks above 20-period high, 1d RSI > 50, volume > 1.5x average
# Short when price breaks below 20-period low, 1d RSI < 50, volume > 1.5x average
# Exit when price crosses opposite Donchian band or RSI crosses 50
# Designed to capture breakouts in trending markets while filtering false signals
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Donchian_Breakout_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_band = high_roll.values
    lower_band = low_roll.values
    
    # Calculate 1d RSI for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # RSI calculation
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align Donchian bands and RSI to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)  # Note: Donchian calculated on LTF, but we align for consistency
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, RSI > 50 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and 
                rsi_aligned[i] > 50 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, RSI < 50 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower band or RSI crosses below 50
            if (close[i] < lower_aligned[i]) or (rsi_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper band or RSI crosses above 50
            if (close[i] > upper_aligned[i]) or (rsi_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals