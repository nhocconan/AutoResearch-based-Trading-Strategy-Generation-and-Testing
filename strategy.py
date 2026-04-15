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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align Donchian channels to 12h
    donchian_high_20_12h = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_12h = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_20_12h = align_htf_to_ltf(prices, df_1d, donchian_mid_20)
    
    # Calculate 12h ATR(14) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_12h[i]) or np.isnan(donchian_low_20_12h[i]) or 
            np.isnan(donchian_mid_20_12h[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h ATR > 0.5% of price
        vol_filter = atr_14_12h[i] > 0.005 * close[i]
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume_ratio[i] > 1.5
        
        # Long conditions:
        # 1. Price above Donchian midline (bullish bias)
        # 2. Price breaks above Donchian high with volume confirmation
        # 3. Volatility and volume filters
        if (close[i] > donchian_mid_20_12h[i] and
            close[i] > donchian_high_20_12h[i] and
            vol_confirm and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below Donchian midline (bearish bias)
        # 2. Price breaks below Donchian low with volume confirmation
        # 3. Volatility and volume filters
        elif (close[i] < donchian_mid_20_12h[i] and
              close[i] < donchian_low_20_12h[i] and
              vol_confirm and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0