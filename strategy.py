#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Williams Fractal
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Daily Williams Fractal (bearish = sell signal, bullish = buy signal) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish (sell) when high[i] is highest of 5 bars (i-2,i-1,i,i+1,i+2)
    # bullish (buy) when low[i] is lowest of 5 bars
    bearish_fractal = np.zeros(len(high_1d))
    bullish_fractal = np.zeros(len(low_1d))
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-1] and high_1d[i] >= high_1d[i-2] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-1] and low_1d[i] <= low_1d[i-2] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Fractals need 2 extra daily bars for confirmation (Williams original)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # === 4h Moving Average for trend filter ===
    close = prices['close'].values
    ma_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ma_34[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ma = ma_34[i]
        vol_ratio_val = vol_ratio[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        if position == 0:
            # Enter long: bullish fractal + price above MA + volume
            if (bullish_fractal_val > 0 and 
                price_close > ma and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal + price below MA + volume
            elif (bearish_fractal_val > 0 and 
                  price_close < ma and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite fractal or MA cross
            if position == 1 and (bearish_fractal_val > 0 or price_close < ma):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (bullish_fractal_val > 0 or price_close > ma):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Fractal_MA_Volume_Filter"
timeframe = "4h"
leverage = 1.0