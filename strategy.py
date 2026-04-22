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
    
    # Load 12h data for Choppiness Index (14) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_h - min_l
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid div by zero
    
    # Align Choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate 4h EMA(50) for trend
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 4h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chop > 61.8 (range) + RSI < 40 + price > EMA50
            if (chop_aligned[i] > 61.8 and 
                rsi[i] < 40 and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: Chop > 61.8 (range) + RSI > 60 + price < EMA50
            elif (chop_aligned[i] > 61.8 and 
                  rsi[i] > 60 and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral (40-60) or Chop < 38.2 (trend)
            if position == 1:
                if (rsi[i] >= 40 and rsi[i] <= 60) or chop_aligned[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (rsi[i] >= 40 and rsi[i] <= 60) or chop_aligned[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Chop618_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0