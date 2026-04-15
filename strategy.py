#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate 20-period weekly Donchian channels
    weekly_highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period weekly EMA for trend filter
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly RSI(14) for momentum filter
    weekly_delta = np.diff(weekly_close, prepend=weekly_close[0])
    weekly_gain = np.where(weekly_delta > 0, weekly_delta, 0)
    weekly_loss = np.where(weekly_delta < 0, -weekly_delta, 0)
    weekly_avg_gain = pd.Series(weekly_gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    weekly_avg_loss = pd.Series(weekly_loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    weekly_rs = weekly_avg_gain / (weekly_avg_loss + 1e-10)
    weekly_rsi_14 = 100 - (100 / (1 + weekly_rs))
    
    # Align HTF indicators to daily timeframe with proper delay
    weekly_ema_50_1d = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    weekly_rsi_14_1d = align_htf_to_ltf(prices, df_1w, weekly_rsi_14)
    weekly_highest_20_1d = align_htf_to_ltf(prices, df_1w, weekly_highest_20)
    weekly_lowest_20_1d = align_htf_to_ltf(prices, df_1w, weekly_lowest_20)
    
    # Calculate daily ATR(14) for volatility filter
    daily_close_prev = np.concatenate([[close[0]], close[:-1]])
    daily_tr = np.maximum(high - low,
                          np.maximum(np.abs(high - daily_close_prev),
                                     np.abs(low - daily_close_prev)))
    daily_atr_14 = pd.Series(daily_tr).rolling(window=14, min_periods=14).mean().values
    daily_atr_ma_50 = pd.Series(daily_atr_14).rolling(window=50, min_periods=50).mean().values
    daily_volatility_ratio = daily_atr_14 / (daily_atr_ma_50 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_50_1d[i]) or np.isnan(weekly_rsi_14_1d[i]) or 
            np.isnan(weekly_highest_20_1d[i]) or np.isnan(weekly_lowest_20_1d[i]) or 
            np.isnan(daily_volatility_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Weekly momentum filter: RSI not extreme (avoid exhaustion)
        # 3. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 4. Weekly Donchian breakout/breakdown
        # 5. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > weekly_ema_50_1d[i] and      # Weekly uptrend filter
            weekly_rsi_14_1d[i] < 70 and           # Not overbought on weekly
            daily_volatility_ratio[i] > 0.8 and    # Avoid low volatility squeezes
            close[i] > weekly_highest_20_1d[i]):   # Weekly Donchian breakout
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < weekly_ema_50_1d[i] and   # Weekly downtrend filter
              weekly_rsi_14_1d[i] > 30 and       # Not oversold on weekly
              daily_volatility_ratio[i] > 0.8 and  # Avoid low volatility squeezes
              close[i] < weekly_lowest_20_1d[i]):  # Weekly Donchian breakdown
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA_RSI_Volume_Donchian_Breakout"
timeframe = "1d"
leverage = 1.0