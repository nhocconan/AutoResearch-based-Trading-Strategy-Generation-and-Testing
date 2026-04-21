#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR for volatility filter and position sizing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Load weekly data ONCE before loop for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: volume / 20-day average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        daily_ema = ema_34_1d_aligned[i]
        weekly_ema = ema_34_1w_aligned[i]
        atr_val = atr_14_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above both daily and weekly EMA, volume spike, moderate volatility
            if (price_close > daily_ema and price_close > weekly_ema and 
                vol_ratio_val > 1.8 and 
                atr_val > 0.005 and atr_val < 0.08):  # ATR between 0.5% and 8% of price
                signals[i] = 0.25
                position = 1
            # Enter short: price below both daily and weekly EMA, volume spike, moderate volatility
            elif (price_close < daily_ema and price_close < weekly_ema and 
                  vol_ratio_val > 1.8 and 
                  atr_val > 0.005 and atr_val < 0.08):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses below/above EMA or volatility too high/low
            if position == 1 and (price_close < daily_ema or atr_val > 0.12 or atr_val < 0.003):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > daily_ema or atr_val > 0.12 or atr_val < 0.003):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyWeeklyEMA34_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0