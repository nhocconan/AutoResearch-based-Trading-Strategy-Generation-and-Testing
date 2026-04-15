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
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 4h
    upper_20_4h = align_htf_to_ltf(prices, df_12h, upper_20_12h)
    lower_20_4h = align_htf_to_ltf(prices, df_12h, lower_20_12h)
    
    # Get 1d HTF data for daily ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR percentage of price (normalized volatility)
    atr_pct_1d = atr_14_1d / close_1d
    
    # Align 1d ATR% to 4h
    atr_pct_1d_4h = align_htf_to_ltf(prices, df_1d, atr_pct_1d)
    
    # Calculate 4h ATR(14) for position sizing scaling
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 4h - full coverage)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 4h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(atr_pct_1d_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 1d ATR% is above 0.8% (avoid low volatility chop)
        vol_regime = atr_pct_1d_4h[i] > 0.008
        
        # Long conditions:
        # 1. 4h price breaks above 12h Donchian upper (20) - bullish breakout
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility regime: ATR% > 0.8% (sufficient volatility for meaningful moves)
        if (close[i] > upper_20_4h[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            # Scale position size by volatility (inverse vol scaling)
            vol_scalar = min(0.008 / atr_pct_1d_4h[i], 1.5)  # Cap at 1.5x
            base_size = 0.25
            signals[i] = base_size * vol_scalar
            
        # Short conditions:
        # 1. 4h price breaks below 12h Donchian lower (20) - bearish breakdown
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility regime: ATR% > 0.8%
        elif (close[i] < lower_20_4h[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            # Scale position size by volatility (inverse vol scaling)
            vol_scalar = min(0.008 / atr_pct_1d_4h[i], 1.5)  # Cap at 1.5x
            base_size = 0.25
            signals[i] = -base_size * vol_scalar
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian20_1d_VolRegime_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0