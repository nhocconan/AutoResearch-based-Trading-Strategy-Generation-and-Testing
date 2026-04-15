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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    h12_close = df_12h['close'].values
    h12_high = df_12h['high'].values
    h12_low = df_12h['low'].values
    h12_volume = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_20 = pd.Series(h12_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(h12_low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Align HTF Donchian channels to 4h timeframe
    highest_20_4h = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_4h = align_htf_to_ltf(prices, df_12h, lowest_20)
    donchian_mid_4h = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_4h[i]) or np.isnan(lowest_20_4h[i]) or 
            np.isnan(donchian_mid_4h[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above 12h Donchian upper with volume confirmation → long (strong continuation)
        # 2. 4h price breaks below 12h Donchian lower with volume confirmation → short (strong continuation)
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        # 6. Trend filter: price above/below Donchian midpoint for directional bias
        
        # Long conditions: 4h breakout above 12h Donchian upper with trend confirmation
        if (close[i] > highest_20_4h[i] and            # 4h price above 12h Donchian upper
            close[i] > donchian_mid_4h[i] and          # Price above midpoint (uptrend bias)
            volume_ratio[i] > 1.3 and                  # Volume confirmation
            atr_14[i] > 0.005 * close[i]):             # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below 12h Donchian lower with trend confirmation
        elif (close[i] < lowest_20_4h[i] and           # 4h price below 12h Donchian lower
              close[i] < donchian_mid_4h[i] and        # Price below midpoint (downtrend bias)
              volume_ratio[i] > 1.3 and                # Volume confirmation
              atr_14[i] > 0.005 * close[i]):           # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian20_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0