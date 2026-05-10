#!/usr/bin/env python3

name = "4H_4C_Reversal_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4-period close price change (4C) for momentum
    price_change_4 = (close - np.roll(close, 4)) / np.roll(close, 4)
    price_change_4[0:4] = np.nan
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 4)  # Warmup for EMA, ATR, and price change
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(price_change_4[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid low volatility regimes
        vol_filter = atr_14_1d_aligned[i] > np.nanmedian(atr_14_1d_aligned[:i+1]) * 0.8
        
        if position == 0:
            # Long entry: negative 4C momentum reversal in uptrend with volume spike
            if (price_change_4[i] < -0.02 and  # Significant 4-period drop
                price_above_ema and 
                volume[i] > vol_threshold[i] and
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: positive 4C momentum reversal in downtrend with volume spike
            elif (price_change_4[i] > 0.02 and   # Significant 4-period rise
                  price_below_ema and 
                  volume[i] > vol_threshold[i] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum reversal or trend breakdown
            if (price_change_4[i] > 0.015 or  # Recovery momentum
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum reversal or trend reversal
            if (price_change_4[i] < -0.015 or  # Continuation of drop
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals