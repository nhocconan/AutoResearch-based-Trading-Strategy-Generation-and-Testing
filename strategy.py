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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h RSI(14) for momentum filter
    delta = np.diff(df_4h['close'].values, prepend=df_4h['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate 4h ATR(14) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    prev_close = np.concatenate([[close_4h[0]], close_4h[:-1]])
    tr = np.maximum(high_4h - low_4h,
                    np.maximum(np.abs(high_4h - prev_close),
                               np.abs(low_4h - prev_close)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / (atr_ma_50 + 1e-10)
    
    # Align HTF indicators to 1h timeframe with proper delay
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50)
    rsi_14_1h = align_htf_to_ltf(prices, df_4h, rsi_14)
    volatility_ratio_1h = align_htf_to_ltf(prices, df_4h, volatility_ratio)
    
    # Calculate 1h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1h[i]) or np.isnan(rsi_14_1h[i]) or 
            np.isnan(volatility_ratio_1h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h trend filter: price above/below 4h EMA50
        # 2. 4h momentum filter: RSI not extreme
        # 3. 4h volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 4. 1h Donchian breakout with volume confirmation
        # 5. Discrete position sizing: 0.20
        
        # Long conditions
        if (close[i] > ema_50_1h[i] and  # Uptrend filter
            rsi_14_1h[i] < 70 and       # Not overbought
            volatility_ratio_1h[i] > 0.8 and  # Avoid low volatility squeezes
            close[i] > highest_20[i] and     # Donchian breakout
            volume_ratio[i] > 1.5):        # Volume confirmation
            signals[i] = 0.20
            
        # Short conditions
        elif (close[i] < ema_50_1h[i] and   # Downtrend filter
              rsi_14_1h[i] > 30 and       # Not oversold
              volatility_ratio_1h[i] > 0.8 and  # Avoid low volatility squeezes
              close[i] < lowest_20[i] and      # Donchian breakdown
              volume_ratio[i] > 1.5):        # Volume confirmation
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4hEMA_RSI_Volume_Donchian_Breakout_Session"
timeframe = "1h"
leverage = 1.0