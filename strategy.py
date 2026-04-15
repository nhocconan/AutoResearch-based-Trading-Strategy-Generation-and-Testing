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
    
    # Get daily HTF data once before loop (12h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1d EMA20 for trend filter
    ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d RSI(14) for momentum filter (avoid extremes)
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 12h timeframe with proper delay
    ema_20_12h = align_htf_to_ltf(prices, df_1d, ema_20)
    rsi_14_12h = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h[i]) or np.isnan(rsi_14_12h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA20
        # 2. 1d momentum filter: RSI not extreme (30-70 range)
        # 3. 12h Donchian breakout: price breaks 20-period high/low
        # 4. 12h volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > ema_20_12h[i] and          # Daily uptrend filter
            rsi_14_12h[i] < 70 and                # Not overbought
            close[i] > highest_20[i] and          # 12h Donchian breakout
            volume_ratio[i] > 1.3):               # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < ema_20_12h[i] and        # Daily downtrend filter
              rsi_14_12h[i] > 30 and              # Not oversold
              close[i] < lowest_20[i] and         # 12h Donchian breakdown
              volume_ratio[i] > 1.3):             # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_EMA20_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0