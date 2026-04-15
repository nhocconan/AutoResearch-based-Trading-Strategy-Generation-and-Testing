#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly EMA21 for trend filter
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate weekly ATR14 for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    weekly_atr14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Align HTF indicators to daily timeframe with proper delay
    weekly_ema21_daily = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    weekly_atr14_daily = align_htf_to_ltf(prices, df_1w, weekly_atr14)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema21_daily[i]) or np.isnan(weekly_atr14_daily[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA21
        # 2. Daily Donchian breakout: price breaks 20-period channel
        # 3. Daily volume confirmation: volume > 1.5x average
        # 4. Weekly volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > weekly_ema21_daily[i] and          # Weekly uptrend filter
            close[i] > highest_20[i] and                 # Donchian breakout
            volume_ratio[i] > 1.5 and                    # Volume confirmation
            weekly_atr14_daily[i] > 0.003 * close[i]):   # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < weekly_ema21_daily[i] and       # Weekly downtrend filter
              close[i] < lowest_20[i] and                # Donchian breakdown
              volume_ratio[i] > 1.5 and                  # Volume confirmation
              weekly_atr14_daily[i] > 0.003 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA21_Donchian20_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0