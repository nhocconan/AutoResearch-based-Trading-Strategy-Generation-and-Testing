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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR14 for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 6h Donchian channels (20-period)
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (focus on active UTC hours: 00-20)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine volatility regime: high volatility when 1d ATR > 30-day average
        if i >= 30:
            atr_ma_30 = pd.Series(atr14_1d_aligned[:i+1]).rolling(window=30, min_periods=30).mean().iloc[-1]
            high_vol_regime = atr14_1d_aligned[i] > 1.2 * atr_ma_30
        else:
            high_vol_regime = False
        
        # Long conditions:
        # 1. 6h price breaks above 20-period Donchian upper
        # 2. Price above 1d EMA34 (bullish trend filter)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. High volatility regime (avoid low volatility chop)
        if (close[i] > upper_20[i] and
            close[i] > ema34_1d_aligned[i] and
            volume_ratio[i] > 1.5 and
            high_vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 20-period Donchian lower
        # 2. Price below 1d EMA34 (bearish trend filter)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. High volatility regime
        elif (close[i] < lower_20[i] and
              close[i] < ema34_1d_aligned[i] and
              volume_ratio[i] > 1.5 and
              high_vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_EMA34_ATR_Volume_Donchian20_Breakout_v1"
timeframe = "6h"
leverage = 1.0