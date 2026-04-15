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
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d RSI(14) for momentum filter
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 12h timeframe with proper delay
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    rsi_14_12h = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h[i]) or np.isnan(rsi_14_12h[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA50
        # 2. 1d momentum filter: RSI in neutral zone (30-70)
        # 3. 12h Donchian breakout: price breaks 20-period channel
        # 4. 12h volume confirmation: volume > 2.0x average
        # 5. 1d volatility filter: ATR > 0 (always true but ensures data validity)
        # 6. Discrete position sizing: 0.30
        
        # Long conditions: break above highest_20 in uptrend
        if (close[i] > ema_50_12h[i] and          # Daily uptrend filter
            30 <= rsi_14_12h[i] <= 70 and         # RSI in neutral zone
            close[i] > highest_20[i] and          # Donchian breakout
            volume_ratio[i] > 2.0):               # Volume confirmation
            signals[i] = 0.30
            
        # Short conditions: break below lowest_20 in downtrend
        elif (close[i] < ema_50_12h[i] and        # Daily downtrend filter
              30 <= rsi_14_12h[i] <= 70 and       # RSI in neutral zone
              close[i] < lowest_20[i] and         # Donchian breakdown
              volume_ratio[i] > 2.0):             # Volume confirmation
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_EMA50_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0