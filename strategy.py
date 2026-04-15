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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h RSI(14) for momentum filter
    delta = pd.Series(df_4h['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # 1h Donchian(20) breakout for entry timing
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_14_4h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 4h ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_4h_aligned[i] > 0.003 * close[i]
        
        # Long conditions:
        # 1. Price above 4h EMA50 (bullish bias)
        # 2. 4h RSI between 45 and 55 (neutral momentum)
        # 3. Volatility filter
        # 4. Price breaks above 1h Donchian high (breakout entry)
        if (close[i] > ema_50_4h_aligned[i] and
            45 <= rsi_14_4h_aligned[i] <= 55 and
            vol_filter and
            close[i] > highest_high[i-1]):  # break above prior 20-bar high
            signals[i] = 0.20
            
        # Short conditions:
        # 1. Price below 4h EMA50 (bearish bias)
        # 2. 4h RSI between 45 and 55 (neutral momentum)
        # 3. Volatility filter
        # 4. Price breaks below 1h Donchian low (breakdown entry)
        elif (close[i] < ema_50_4h_aligned[i] and
              45 <= rsi_14_4h_aligned[i] <= 55 and
              vol_filter and
              close[i] < lowest_low[i-1]):  # break below prior 20-bar low
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA50_RSI14_Vol_Donchian20_v1"
timeframe = "1h"
leverage = 1.0