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
    
    # Calculate 12h ATR(14) for volatility regime
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_12h = pd.Series(atr_14_12h).rolling(window=50, min_periods=50).mean().values
    atr_ratio_12h = atr_14_12h / (atr_ma_50_12h + 1e-10)
    
    # Align 12h ATR ratio to 6h
    atr_ratio_6h = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    # Get 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC
    prev_high = pd.Series(high_1d).shift(1).values
    prev_low = pd.Series(low_1d).shift(1).values
    prev_close = pd.Series(close_1d).shift(1).values
    
    # Camarilla calculations
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_h3 = camarilla_pivot + camarilla_range * 1.1 / 4.0
    camarilla_l3 = camarilla_pivot - camarilla_range * 1.1 / 4.0
    camarilla_h4 = camarilla_pivot + camarilla_range * 1.1 / 2.0
    camarilla_l4 = camarilla_pivot - camarilla_range * 1.1 / 2.0
    
    # Align Camarilla levels to 6h
    camarilla_h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 6h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_6h[i]) or np.isnan(camarilla_h3_6h[i]) or 
            np.isnan(camarilla_l3_6h[i]) or np.isnan(camarilla_h4_6h[i]) or 
            np.isnan(camarilla_l4_6h[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above Camarilla H3 with volume
        # 2. ATR ratio > 0.8 (avoid extremely low volatility)
        # 3. ATR ratio < 2.0 (avoid extremely high volatility spikes)
        if (close[i] > camarilla_h3_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_ratio_6h[i] > 0.8 and
            atr_ratio_6h[i] < 2.0):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below Camarilla L3 with volume
        # 2. ATR ratio > 0.8 (avoid extremely low volatility)
        # 3. ATR ratio < 2.0 (avoid extremely high volatility spikes)
        elif (close[i] < camarilla_l3_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_ratio_6h[i] > 0.8 and
              atr_ratio_6h[i] < 2.0):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_ATR_Regime_1d_CamarillaH3L3_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0