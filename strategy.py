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
    
    # Calculate 12h Donchian(20) channels
    donch_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Calculate 4h ATR(14) for volatility filter and stoploss
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
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above 12h Donchian high with volume confirmation → long
        # 2. 4h price breaks below 12h Donchian low with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above 12h Donchian high
        if (close[i] > donch_high_4h[i] and            # 4h price above 12h Donchian high
            volume_ratio[i] > 1.5 and                  # Volume confirmation
            atr_14[i] > 0.005 * close[i]):             # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below 12h Donchian low
        elif (close[i] < donch_low_4h[i] and           # 4h price below 12h Donchian low
              volume_ratio[i] > 1.5 and                # Volume confirmation
              atr_14[i] > 0.005 * close[i]):           # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian20_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0