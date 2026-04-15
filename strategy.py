#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) with volume confirmation and ATR filter.
# Camarilla pivots identify intraday support/resistance; breaks of R3/S3 indicate strong momentum.
# Volume confirms institutional participation. ATR filter avoids low-volatility false breakouts.
# Works in bull/bear: breakouts capture trends; volume/ATR filter reduces whipsaws.
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.

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
    
    # Calculate 1d Camarilla levels (based on prior day)
    # Camarilla: 
    # H4 = Close + 1.1*(High-Low)/2
    # L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4
    # L3 = Close - 1.1*(High-Low)/4
    # H2 = Close + 1.1*(High-Low)/6
    # L2 = Close - 1.1*(High-Low)/6
    # H1 = Close + 1.1*(High-Low)/12
    # L1 = Close - 1.1*(High-Low)/12
    # Pivot = (High + Low + Close)/3
    # We use H3/L3 for strong breakouts (equivalent to R3/S3 in some definitions)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's OHLC
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    
    # Camarilla levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align to 4h
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
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_4h[i]) or np.isnan(camarilla_l3_4h[i]) or 
            np.isnan(camarilla_h4_4h[i]) or np.isnan(camarilla_l4_4h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above Camarilla H3 (strong intraday resistance)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > camarilla_h3_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below Camarilla L3 (strong intraday support)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < camarilla_l3_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_CamarillaH3L3_Volume_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0