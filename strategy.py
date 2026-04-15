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
    
    # Get weekly HTF data once before loop (6h primary, 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1_w = pd.Series(weekly_high - weekly_low)
    tr2_w = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3_w = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
    atr_14_w = tr_w.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_34_w_6h = align_htf_to_ltf(prices, df_1w, ema_34_w)
    atr_14_w_6h = align_htf_to_ltf(prices, df_1w, atr_14_w)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_w_6h[i]) or np.isnan(atr_14_w_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA34
        # 2. 6h Donchian breakout: price breaks 20-period channel
        # 3. 6h volume confirmation: volume > 1.5x average (moderate filter)
        # 4. Weekly volatility filter: ATR > 0.003 * price (avoid extremely low volatility)
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in weekly uptrend
        if (close[i] > ema_34_w_6h[i] and          # Weekly uptrend filter
            close[i] > highest_20[i] and          # Donchian breakout
            volume_ratio[i] > 1.5 and             # Volume confirmation
            atr_14_w_6h[i] > 0.003 * close[i]):   # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in weekly downtrend
        elif (close[i] < ema_34_w_6h[i] and        # Weekly downtrend filter
              close[i] < lowest_20[i] and          # Donchian breakdown
              volume_ratio[i] > 1.5 and            # Volume confirmation
              atr_14_w_6h[i] > 0.003 * close[i]):  # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA34_Donchian20_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0