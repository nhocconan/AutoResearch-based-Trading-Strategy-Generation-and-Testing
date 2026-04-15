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
    
    # Calculate weekly Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = weekly_volume / (vol_ma_20 + 1e-10)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly EMA50 for trend filter
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to daily timeframe with proper delay
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50)
    atr_14_1d = align_htf_to_ltf(prices, df_1w, atr_14)
    highest_20_1d = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_1d = align_htf_to_ltf(prices, df_1w, lowest_20)
    volume_ratio_1d = align_htf_to_ltf(prices, df_1w, volume_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d[i]) or np.isnan(atr_14_1d[i]) or 
            np.isnan(highest_20_1d[i]) or np.isnan(lowest_20_1d[i]) or np.isnan(volume_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Weekly Donchian breakout: price breaks 20-period channel
        # 3. Weekly volume confirmation: volume > 1.5x average (moderate filter)
        # 4. Weekly volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > ema_50_1d[i] and          # Weekly uptrend filter
            close[i] > highest_20_1d[i] and      # Donchian breakout
            volume_ratio_1d[i] > 1.5 and         # Volume confirmation
            atr_14_1d[i] > 0.003 * close[i]):    # Volatility filter (ATR > 0.3% of price)
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < ema_50_1d[i] and        # Weekly downtrend filter
              close[i] < lowest_20_1d[i] and     # Donchian breakdown
              volume_ratio_1d[i] > 1.5 and       # Volume confirmation
              atr_14_1d[i] > 0.003 * close[i]):  # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Donchian_Breakout_EMA50_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0