#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and ATR filter
# Uses daily pivots for structure, 12h for execution. Designed for low trade frequency
# (<30/year) to avoid fee drag. Works in bull/bear via breakouts with volume validation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily Camarilla pivot levels
    # Camarilla: P = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using close-based Camarilla for intraday relevance
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_hl = daily_high - daily_low
    r1 = daily_close + range_hl * 1.1 / 12.0
    s1 = daily_close - range_hl * 1.1 / 12.0
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h Donchian breakout (20-period) for confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 12h close above R1 with volume confirmation and volatility
        # Short: 12h close below S1 with volume confirmation and volatility
        # Discrete size: 0.25 to manage drawdown
        
        # Long conditions
        if (close[i] > r1_12h[i] and            # 12h price above R1 Camarilla
            volume_ratio[i] > 1.5 and          # Strong volume confirmation
            atr_14_12h[i] > 0.003 * close[i]): # Minimum volatility filter
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < s1_12h[i] and          # 12h price below S1 Camarilla
              volume_ratio[i] > 1.5 and         # Strong volume confirmation
              atr_14_12h[i] > 0.003 * close[i]): # Minimum volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0