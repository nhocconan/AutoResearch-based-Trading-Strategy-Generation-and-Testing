#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: volume / 20-day average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        daily_ema = ema_34_1d_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above daily EMA, moderate volatility, volume spike
            if (price_close > daily_ema and 
                vol_ratio_val > 1.5 and 
                atr_ratio_val > 0.8 and atr_ratio_val < 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below daily EMA, moderate volatility, volume spike
            elif (price_close < daily_ema and 
                  vol_ratio_val > 1.5 and 
                  atr_ratio_val > 0.8 and atr_ratio_val < 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse crossover or volatility extremes
            if position == 1 and (price_close < daily_ema or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > daily_ema or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_DailyEMA34_Volume_ATR_Filter_v2"
timeframe = "1d"
leverage = 1.0