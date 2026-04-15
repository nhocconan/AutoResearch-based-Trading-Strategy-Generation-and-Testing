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
    
    # Get daily HTF data once before loop (4h primary, 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
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
    
    # Align HTF indicators to 4h timeframe with proper delay
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    rsi_14_4h = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h[i]) or np.isnan(rsi_14_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d trend filter: price above/below daily EMA50
        # 2. 1d momentum filter: RSI not extreme (avoid overbought/oversold)
        # 3. 4h Donchian breakout: price breaks 20-period channel
        # 4. 4h volatility filter: ATR > 0 (always true, but ensures we have volatility data)
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend
        if (close[i] > ema_50_4h[i] and          # Daily uptrend filter
            rsi_14_4h[i] < 70 and                # Not overbought
            close[i] > highest_20[i]):           # Donchian breakout
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend
        elif (close[i] < ema_50_4h[i] and        # Daily downtrend filter
              rsi_14_4h[i] > 30 and              # Not oversold
              close[i] < lowest_20[i]):          # Donchian breakdown
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyEMA50_RSI14_Donchian20_Breakout"
timeframe = "4h"
leverage = 1.0