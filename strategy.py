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
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 35  # Need Donchian (20) + weekly EMA21 (21)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema21_1w_aligned[i]
        downtrend = close[i] < ema21_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: break above Donchian high in uptrend with sufficient volatility
            if uptrend and close[i] > high_roll[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in downtrend with sufficient volatility
            elif downtrend and close[i] < low_roll[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below Donchian low or trend reversal
            if close[i] < low_roll[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above Donchian high or trend reversal
            if close[i] > high_roll[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_WeeklyEMA21Trend_VolatilityFilter"
timeframe = "4h"
leverage = 1.0