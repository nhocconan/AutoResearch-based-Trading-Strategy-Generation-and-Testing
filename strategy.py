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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily RSI(14) for mean reversion signals
    delta = pd.Series(daily_close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate daily EMA(200) for trend filter
    ema_200 = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe with proper delay
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_values)
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h[i]) or np.isnan(ema_200_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily RSI extreme (oversold <30 or overbought >70)
        # 2. 4h Donchian breakout in direction of RSI signal
        # 3. Daily trend filter: price above/below daily EMA200
        # 4. 4h volume confirmation: volume > 1.5x average
        # 5. 4h volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: RSI oversold (<30) + Donchian breakout above in uptrend
        if (rsi_4h[i] < 30 and          # Daily oversold
            close[i] > ema_200_4h[i] and         # Daily uptrend filter
            close[i] > highest_20[i] and        # 4h Donchian breakout
            volume_ratio[i] > 1.5 and           # Volume confirmation
            atr_14_4h[i] > 0.003 * close[i]):   # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: RSI overbought (>70) + Donchian breakdown below in downtrend
        elif (rsi_4h[i] > 70 and        # Daily overbought
              close[i] < ema_200_4h[i] and       # Daily downtrend filter
              close[i] < lowest_20[i] and       # 4h Donchian breakdown
              volume_ratio[i] > 1.5 and         # Volume confirmation
              atr_14_4h[i] > 0.003 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_RSI_Donchian_Breakout_EMA200_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0