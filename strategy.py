#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with volume confirmation and weekly ATR filter
# Uses 1w ATR to avoid low volatility regimes and 1d Camarilla levels for structure
# Discrete position sizing (0.25) to minimize fee drag. Designed for BTC/ETH in both bull/bear markets.
# Target: 12-37 trades/year (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3, L3, H4, L4)
    daily_range = daily_high - daily_low
    camarilla_h3 = daily_close + 1.1 * daily_range / 2
    camarilla_l3 = daily_close - 1.1 * daily_range / 2
    camarilla_h4 = daily_close + 1.1 * daily_range
    camarilla_l4 = daily_close - 1.1 * daily_range
    
    # Get weekly HTF data once before loop for ATR filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_w = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    camarilla_h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_14_w_12h = align_htf_to_ltf(prices, df_1w, atr_14_w)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_12h[i]) or np.isnan(camarilla_l3_12h[i]) or 
            np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(atr_14_w_12h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h price breaks above H3 with volume confirmation → long
        # 2. 12h price breaks below L3 with volume confirmation → short
        # 3. Volatility filter: weekly ATR > 1% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 12h breakout above H3
        if (close[i] > camarilla_h3_12h[i] and            # 12h price above H3
            volume_ratio[i] > 1.5 and                     # Volume confirmation
            atr_14_w_12h[i] > 0.01 * close[i]):           # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 12h breakdown below L3
        elif (close[i] < camarilla_l3_12h[i] and          # 12h price below L3
              volume_ratio[i] > 1.5 and                   # Volume confirmation
              atr_14_w_12h[i] > 0.01 * close[i]):         # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_H3_L3_Breakout_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0