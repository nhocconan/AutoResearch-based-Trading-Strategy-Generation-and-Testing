#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Daily ATR (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Daily Volume MA (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Weekly EMA (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        atr = atr_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        ema = ema_21_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: Price above weekly EMA21, volume spike, and volatility expansion
            if (price > ema and 
                vol > 1.8 * vol_ma and 
                atr > 1.2 * pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA21, volume spike, and volatility expansion
            elif (price < ema and 
                  vol > 1.8 * vol_ma and 
                  atr > 1.2 * pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values[i]):
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: Price crosses below weekly EMA21 OR volatility contraction
            if price < ema or atr < 0.8 * pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly EMA21 OR volatility contraction
            if price > ema or atr < 0.8 * pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA21_VolumeVolatilityFilter"
timeframe = "12h"
leverage = 1.0