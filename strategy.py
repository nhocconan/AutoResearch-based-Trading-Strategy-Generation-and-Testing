#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1-day ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1-day ADX for trend strength filter
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1-day EMA200 for long-term trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 12-period ATR for 12h timeframe volatility
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]
    atr_12h = pd.Series(tr_12h).rolling(window=12, min_periods=12).mean().values
    
    # Volatility filter: current volatility > 1.5x average volatility
    vol_filter = atr_12h > (pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above EMA50 and EMA200, strong trend (ADX > 25), volatility filter
        if (close[i] > ema50_1d_aligned[i] and 
            close[i] > ema200_1d_aligned[i] and 
            adx_1d_aligned[i] > 25 and 
            vol_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below EMA50 and EMA200, strong trend (ADX > 25), volatility filter
        elif (close[i] < ema50_1d_aligned[i] and 
              close[i] < ema200_1d_aligned[i] and 
              adx_1d_aligned[i] > 25 and 
              vol_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend weakens (ADX < 20) or reversal signals
        elif position == 1 and (adx_1d_aligned[i] < 20 or close[i] < ema50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (adx_1d_aligned[i] < 20 or close[i] > ema50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_EMA50_200_ADX25_VolatilityFilter"
timeframe = "12h"
leverage = 1.0