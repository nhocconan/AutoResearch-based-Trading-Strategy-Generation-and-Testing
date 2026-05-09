#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Daily Camarilla levels: R1, S1
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    # Camarilla: R1 = close + 1.1*range/12, S1 = close - 1.1*range/12
    camarilla_R1 = close_1d + 1.1 * range_1d / 12
    camarilla_S1 = close_1d - 1.1 * range_1d / 12
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1, additional_delay_bars=1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1, additional_delay_bars=1)
    
    # 4h ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA50, with volume
            if (price > camarilla_R1_aligned[i] and 
                price > ema50_12h_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1, below 12h EMA50, with volume
            elif (price < camarilla_S1_aligned[i] and 
                  price < ema50_12h_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses below S1 or loses volume
            if (price < camarilla_S1_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R1 or loses volume
            if (price > camarilla_R1_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals