#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 12h high/low for Donchian channel (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian channels (20-period)
    high_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_12h, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_12h, low_min)
    
    # 4h volume moving average for confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ATR for volatility filter and position sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(high_max_aligned[i]) or 
            np.isnan(low_min_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        # Volume confirmation: above average volume
        vol_confirmed = vol > vol_ma_val
        
        if position == 0 and vol_confirmed:
            # Long: price breaks above 12h Donchian upper band
            if price > high_max_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band
            elif price < low_min_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to the opposite Donchian band
            if position == 1 and price < low_min_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > high_max_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_12hVolFilter_v1"
timeframe = "4h"
leverage = 1.0