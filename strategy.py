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
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0) for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for 4h
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend components
    hl2_4h = (high_4h + low_4h) / 2
    upper_basic_4h = hl2_4h + (3.0 * atr_4h)
    lower_basic_4h = hl2_4h - (3.0 * atr_4h)
    
    upper_band_4h = np.zeros_like(close_4h)
    lower_band_4h = np.zeros_like(close_4h)
    
    for i in range(len(close_4h)):
        if i == 0:
            upper_band_4h[i] = upper_basic_4h[i]
            lower_band_4h[i] = lower_basic_4h[i]
        else:
            if upper_basic_4h[i] < upper_band_4h[i-1] or close_4h[i-1] > upper_band_4h[i-1]:
                upper_band_4h[i] = upper_basic_4h[i]
            else:
                upper_band_4h[i] = upper_band_4h[i-1]
                
            if lower_basic_4h[i] > lower_band_4h[i-1] or close_4h[i-1] < lower_band_4h[i-1]:
                lower_band_4h[i] = lower_basic_4h[i]
            else:
                lower_band_4h[i] = lower_band_4h[i-1]
    
    supertrend_4h = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i == 0:
            supertrend_4h[i] = upper_band_4h[i]
        else:
            if supertrend_4h[i-1] == upper_band_4h[i-1] and close_4h[i] <= upper_band_4h[i]:
                supertrend_4h[i] = upper_band_4h[i]
            elif supertrend_4h[i-1] == upper_band_4h[i-1] and close_4h[i] > upper_band_4h[i]:
                supertrend_4h[i] = lower_band_4h[i]
            elif supertrend_4h[i-1] == lower_band_4h[i-1] and close_4h[i] >= lower_band_4h[i]:
                supertrend_4h[i] = lower_band_4h[i]
            elif supertrend_4h[i-1] == lower_band_4h[i-1] and close_4h[i] < lower_band_4h[i]:
                supertrend_4h[i] = upper_band_4h[i]
    
    # Align 4h Supertrend to 1h
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    
    # Calculate 1h Donchian channels (20-period)
    high_1h = high
    low_1h = low
    upper_20_1h = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    lower_20_1h = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h ATR(14) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(upper_20_1h[i]) or 
            np.isnan(lower_20_1h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price breaks above 1h Donchian upper (20)
        # 2. 4h Supertrend is bullish (price > Supertrend)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price (avoid low volatility chop)
        if (close[i] > upper_20_1h[i] and
            close[i] > supertrend_4h_aligned[i] and
            volume_ratio[i] > 1.3 and
            atr_14[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1h price breaks below 1h Donchian lower (20)
        # 2. 4h Supertrend is bearish (price < Supertrend)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price
        elif (close[i] < lower_20_1h[i] and
              close[i] < supertrend_4h_aligned[i] and
              volume_ratio[i] > 1.3 and
              atr_14[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_Supertrend10_3_Donchian20_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0