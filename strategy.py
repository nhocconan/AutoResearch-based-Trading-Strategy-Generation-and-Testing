#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    mid_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = pd.Series(df_4h['high'] - df_4h['low'])
    tr2 = pd.Series(np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]])))
    tr3 = pd.Series(np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe with proper delay
    highest_20_1h = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_1h = align_htf_to_ltf(prices, df_4h, lowest_20)
    mid_20_1h = align_htf_to_ltf(prices, df_4h, mid_20)
    atr_14_1h = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_1h[i]) or np.isnan(lowest_20_1h[i]) or 
            np.isnan(mid_20_1h[i]) or np.isnan(atr_14_1h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: price breaks above 4h Donchian upper band with volume confirmation
        # Short: price breaks below 4h Donchian lower band with volume confirmation
        # Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # Volume confirmation: volume > 1.5x average
        # Discrete position sizing: 0.20
        
        # Long conditions: breakout above 4h Donchian upper band
        if (close[i] > highest_20_1h[i] and            # price above 4h upper band
            volume_ratio[i] > 1.5 and                  # volume confirmation
            atr_14_1h[i] > 0.005 * close[i]):          # volatility filter
            signals[i] = 0.20
            
        # Short conditions: breakdown below 4h Donchian lower band
        elif (close[i] < lowest_20_1h[i] and           # price below 4h lower band
              volume_ratio[i] > 1.5 and                # volume confirmation
              atr_14_1h[i] > 0.005 * close[i]):        # volatility filter
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian_Breakout_Volume_ATR_Filter_Session"
timeframe = "1h"
leverage = 1.0