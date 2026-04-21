#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align daily indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: volume / 20-period average volume (12h)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_ema = ema_34_1w_aligned[i]
        daily_ema = ema_50_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above both EMAs, volume spike, moderate volatility
            if (price_close > weekly_ema and price_close > daily_ema and 
                vol_ratio_val > 1.8 and 
                atr_val > 0.5 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1]) and
                atr_val < 2.0 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = 0.25
                position = 1
            # Enter short: price below both EMAs, volume spike, moderate volatility
            elif (price_close < weekly_ema and price_close < daily_ema and 
                  vol_ratio_val > 1.8 and 
                  atr_val > 0.5 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1]) and
                  atr_val < 2.0 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse crossover of either EMA or volatility extremes
            if position == 1 and (price_close < weekly_ema or price_close < daily_ema):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > weekly_ema or price_close > daily_ema):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DualEMA_Volume_Filter"
timeframe = "12h"
leverage = 1.0