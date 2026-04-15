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
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h timeframe (primary)
    upper_20_4h = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Get 12h HTF data for regime filter (choppiness)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Chopiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(TR) / (HH - LL)) / log10(14)
    sum_tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    chop_denom = hh_14 - ll_14
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_12h = 100 * np.log10(sum_tr_14 / chop_denom_safe) / np.log10(14)
    
    # Align Chopiness Index to 4h
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if 'close_4h' in locals() else np.abs(high_4h - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if 'close_4h' in locals() else np.abs(low_4h - np.concatenate([[close[0]], close[:-1]]))
    # Recalculate properly for 4h close
    df_4h_close = df_4h['close'].values
    tr2_4h = np.abs(high_4h - np.concatenate([[df_4h_close[0]], df_4h_close[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[df_4h_close[0]], df_4h_close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ratio_4h = df_4h['volume'].values / (vol_ma_20_4h + 1e-10)
    volume_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ratio_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(chop_12h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(volume_ratio_4h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20)
        # 2. Chopiness regime: CHOP > 61.8 (ranging market) for mean reversion
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        if (close[i] > upper_20_4h[i] and
            chop_12h_aligned[i] > 61.8 and
            volume_ratio_4h_aligned[i] > 1.5 and
            atr_14_4h_aligned[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. Chopiness regime: CHOP > 61.8 (ranging market) for mean reversion
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < lower_20_4h[i] and
              chop_12h_aligned[i] > 61.8 and
              volume_ratio_4h_aligned[i] > 1.5 and
              atr_14_4h_aligned[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Chop618_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0