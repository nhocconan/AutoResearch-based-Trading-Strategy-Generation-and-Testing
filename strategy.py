#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate weekly ATR(14) for volatility regime
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr = np.maximum(weekly_high - weekly_low,
                    np.maximum(np.abs(weekly_high - weekly_close_prev),
                               np.abs(weekly_low - weekly_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_26 = pd.Series(atr_14).rolling(window=26, min_periods=26).mean().values
    volatility_ratio = atr_14 / (atr_ma_26 + 1e-10)
    
    # Calculate weekly RSI(14) for momentum filter
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(21) for trend filter
    ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_21_6h = align_htf_to_ltf(prices, df_1w, ema_21)
    rsi_14_6h = align_htf_to_ltf(prices, df_1w, rsi_14)
    volatility_ratio_6h = align_htf_to_ltf(prices, df_1w, volatility_ratio)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_6h[i]) or np.isnan(rsi_14_6h[i]) or 
            np.isnan(volatility_ratio_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA21
        # 2. Weekly momentum filter: RSI not extreme
        # 3. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 4. 6h Donchian breakout with volume confirmation
        # 5. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_21_6h[i] and  # Uptrend filter
            rsi_14_6h[i] < 70 and       # Not overbought
            volatility_ratio_6h[i] > 0.8 and  # Avoid low volatility squeezes
            close[i] > highest_20[i] and     # Donchian breakout
            volume_ratio[i] > 1.5):        # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_21_6h[i] and   # Downtrend filter
              rsi_14_6h[i] > 30 and       # Not oversold
              volatility_ratio_6h[i] > 0.8 and  # Avoid low volatility squeezes
              close[i] < lowest_20[i] and      # Donchian breakdown
              volume_ratio[i] > 1.5):        # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA21_RSI_Volume_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0