#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 1-year high/low
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 52-week high and low
    high_52w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    
    # Align to daily timeframe
    high_52w_daily = align_htf_to_ltf(prices, df_1w, high_52w)
    low_52w_daily = align_htf_to_ltf(prices, df_1w, low_52w)
    
    # Daily ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 52-week high/low and ATR MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_52w_daily[i]) or 
            np.isnan(low_52w_daily[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        
        if position == 0:
            # Long: price breaks above 52-week high with volume and volatility
            if close[i] > high_52w_daily[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 52-week low with volume and volatility
            elif close[i] < low_52w_daily[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 52-week low or volatility drops
            if close[i] < low_52w_daily[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 52-week high or volatility drops
            if close[i] > high_52w_daily[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_52Week_High_Low_Breakout_Volume_Volatility"
timeframe = "1d"
leverage = 1.0