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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    h12_close = df_12h['close'].values
    h12_high = df_12h['high'].values
    h12_low = df_12h['low'].values
    h12_volume = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period) for trend filter
    highest_20_12h = pd.Series(h12_high).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(h12_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = pd.Series(h12_high - h12_low)
    tr2 = pd.Series(np.abs(h12_high - np.concatenate([[h12_close[0]], h12_close[:-1]])))
    tr3 = pd.Series(np.abs(h12_low - np.concatenate([[h12_close[0]], h12_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    highest_20_12h_6h = align_htf_to_ltf(prices, df_12h, highest_20_12h)
    lowest_20_12h_6h = align_htf_to_ltf(prices, df_12h, lowest_20_12h)
    atr_14_12h_6h = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 6h Donchian channels (10-period) for entry timing
    highest_10_6h = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10_6h = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20_6h + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_12h_6h[i]) or np.isnan(lowest_20_12h_6h[i]) or 
            np.isnan(atr_14_12h_6h[i]) or np.isnan(highest_10_6h[i]) or np.isnan(lowest_10_6h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h ATR > 1.5% of price (avoid low volatility chop)
        volatility_regime = atr_14_12h_6h[i] > 0.015 * close[i]
        
        # Entry conditions with discrete sizing 0.25
        # Long: 6h breakout above 10-period high + 12h uptrend + volume + volatility regime
        if (close[i] > highest_10_6h[i] and            # 6h breakout above recent high
            close[i] > highest_20_12h_6h[i] and       # 12h uptrend: price above 12h 20-period high
            volume_ratio[i] > 1.4 and                 # Volume confirmation
            volatility_regime):                       # Sufficient volatility
            signals[i] = 0.25
            
        # Short: 6h breakdown below 10-period low + 12h downtrend + volume + volatility regime
        elif (close[i] < lowest_10_6h[i] and          # 6h breakdown below recent low
              close[i] < lowest_20_12h_6h[i] and      # 12h downtrend: price below 12h 20-period low
              volume_ratio[i] > 1.4 and               # Volume confirmation
              volatility_regime):                     # Sufficient volatility
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_Donchian_Breakout_Trend_Filter_Volume"
timeframe = "6h"
leverage = 1.0