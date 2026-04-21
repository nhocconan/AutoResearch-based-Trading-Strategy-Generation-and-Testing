#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === Daily Williams %R (14) ===
    high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    close_1d = df_1d['close'].values
    williams_r = -100 * (high_14.values - close_1d) / (high_14.values - low_14.values)
    williams_r[high_14.values == low_14.values] = -50  # avoid div by zero
    
    # === Daily EMA21 for trend filter ===
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        wr = williams_r_aligned[i]
        ema_trend = ema_21_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long when oversold + above EMA21 + volume
            if (wr < -80 and  # Oversold
                price_close > ema_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short when overbought + below EMA21 + volume
            elif (wr > -20 and   # Overbought
                  price_close < ema_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when opposite condition or momentum shifts
            if position == 1 and (wr > -20 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (wr < -80 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_EMA21_Volume"
timeframe = "6h"
leverage = 1.0