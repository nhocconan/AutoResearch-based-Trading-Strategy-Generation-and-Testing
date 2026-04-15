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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels from prior day
    # Prior day high/low/close
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_h3 = camarilla_pivot + (prior_high - prior_low) * 1.1 / 4.0
    camarilla_l3 = camarilla_pivot - (prior_high - prior_low) * 1.1 / 4.0
    camarilla_h4 = camarilla_pivot + (prior_high - prior_low) * 1.1 / 2.0
    camarilla_l4 = camarilla_pivot - (prior_high - prior_low) * 1.1 / 2.0
    
    # Align 1d Camarilla to 4h
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
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
    
    # Precompute session filter (00-24 UTC for 4h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 4h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pivot_4h[i]) or np.isnan(camarilla_h3_4h[i]) or 
            np.isnan(camarilla_l3_4h[i]) or np.isnan(camarilla_h4_4h[i]) or 
            np.isnan(camarilla_l4_4h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price touches or breaks above Camarilla H3 (bullish bias)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if (close[i] >= camarilla_h3_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price touches or breaks below Camarilla L3 (bearish bias)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.3% of price
        elif (close[i] <= camarilla_l3_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_CamarillaH3L3_Volume_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0