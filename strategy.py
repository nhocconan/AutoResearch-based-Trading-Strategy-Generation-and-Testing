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
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA200 for trend filter
    ema_200 = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    ema_200_12h = align_htf_to_ltf(prices, df_1d, ema_200)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_200_12h[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA200 (strong trend)
        # 2. 12h Donchian breakout: price breaks 20-period channel
        # 3. 12h volume confirmation: volume > 1.8x average (slightly relaxed from 2.0)
        # 4. 12h volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in strong uptrend
        if (close[i] > ema_200_12h[i] and          # Daily strong uptrend filter
            close[i] > highest_20[i] and          # Donchian breakout
            volume_ratio[i] > 1.8 and             # Volume confirmation (relaxed from 2.0)
            atr_14_12h[i] > 0.003 * close[i]):    # Volatility filter (ATR > 0.3% of price)
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in strong downtrend
        elif (close[i] < ema_200_12h[i] and        # Daily strong downtrend filter
              close[i] < lowest_20[i] and         # Donchian breakdown
              volume_ratio[i] > 1.8 and           # Volume confirmation
              atr_14_12h[i] > 0.003 * close[i]):  # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_EMA200_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0