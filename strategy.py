#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily true range for ATR
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 14-day RSI
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Align daily ATR and RSI to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h price channels: Donchian(20)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0
        vol_filter = atr_1d_aligned[i] > 0
        
        # Momentum filter: RSI between 30 and 70 (avoid extremes)
        mom_filter = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Volume filter: above average volume
        vol_filter = vol_filter and (volume[i] > vol_ma[i])
        
        # Entry conditions: 
        # Long: price breaks above Donchian high with volume and momentum
        # Short: price breaks below Donchian low with volume and momentum
        long_entry = (close[i] > high_roll[i]) and vol_filter and mom_filter
        short_entry = (close[i] < low_roll[i]) and vol_filter and mom_filter
        
        # Exit conditions: ATR-based stop loss
        long_exit = False
        short_exit = False
        
        if position == 1:
            # Long exit: price drops 2*ATR from entry
            # We approximate by checking if price is below the Donchian low
            long_exit = close[i] < low_roll[i]
        elif position == -1:
            # Short exit: price rises 2*ATR from entry
            # We approximate by checking if price is above the Donchian high
            short_exit = close[i] > high_roll[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_RSI_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0