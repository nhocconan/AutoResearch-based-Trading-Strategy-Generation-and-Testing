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
    
    # Get 12h HTF data once before loop (as per experiment spec)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h RSI(14) for momentum filter
    delta = pd.Series(df_12h['close'].values).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_12h = 100 - (100 / (1 + rs))
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # Calculate 4h Donchian(20) breakout levels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_12h_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(rsi_14_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h ATR is elevated (> 0.5% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.005 * close[i]
        
        # Volume confirmation: require above average volume
        vol_confirm = vol_ratio[i] > 1.2
        
        # Long conditions:
        # 1. Price breaks above 4h Donchian upper band (breakout)
        # 2. Price above 12h EMA34 (bullish bias)
        # 3. 12h RSI > 50 (bullish momentum)
        # 4. Volatility filter
        # 5. Volume confirmation
        if (close[i] > highest_high[i] and 
            close[i] > ema_34_12h_aligned[i] and 
            rsi_14_12h_aligned[i] > 50 and 
            vol_filter and 
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 4h Donchian lower band (breakdown)
        # 2. Price below 12h EMA34 (bearish bias)
        # 3. 12h RSI < 50 (bearish momentum)
        # 4. Volatility filter
        # 5. Volume confirmation
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_34_12h_aligned[i] and 
              rsi_14_12h_aligned[i] < 50 and 
              vol_filter and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA34_RSI_VolFilter_12h"
timeframe = "4h"
leverage = 1.0