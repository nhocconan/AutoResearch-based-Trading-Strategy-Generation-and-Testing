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
    
    # Get weekly HTF data once before loop (1d primary, 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate 1d Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1w EMA21 for trend filter
    ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1w RSI(14) for momentum filter
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 1d timeframe with proper delay
    ema_21_1d = align_htf_to_ltf(prices, df_1w, ema_21)
    rsi_14_1d = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d[i]) or np.isnan(rsi_14_1d[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1w trend filter: price above/below weekly EMA21
        # 2. 1w momentum filter: RSI not extreme (avoid overbought/oversold)
        # 3. 1d Donchian breakout: price breaks 20-period high/low
        # 4. 1d volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > ema_21_1d[i] and          # Weekly uptrend filter
            rsi_14_1d[i] < 70 and                # Not overbought
            close[i] > highest_20[i] and         # Donchian breakout
            volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < ema_21_1d[i] and        # Weekly downtrend filter
              rsi_14_1d[i] > 30 and              # Not oversold
              close[i] < lowest_20[i] and        # Donchian breakdown
              volume_ratio[i] > 1.5):            # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA21_RSI14_Donchian20_Breakout_Volume"
timeframe = "1d"
leverage = 1.0