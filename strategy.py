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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend strength filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_dm = np.where((df_12h['high'] - df_12h['high'].shift(1)) > (df_12h['low'].shift(1) - df_12h['low']), 
                       np.maximum(df_12h['high'] - df_12h['high'].shift(1), 0), 0)
    minus_dm = np.where((df_12h['low'].shift(1) - df_12h['low']) > (df_12h['high'] - df_12h['high'].shift(1)), 
                        np.maximum(df_12h['low'].shift(1) - df_12h['low'], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm[1:]])
    minus_dm = np.concatenate([[0], minus_dm[1:]])
    tr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_14_12h)
    
    # Calculate 12h RSI(14) for overbought/oversold filter
    delta = np.diff(df_12h['close'].values, prepend=df_12h['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_12h = 100 - (100 / (1 + rs))
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # Calculate 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_14_12h_aligned[i]) or np.isnan(rsi_14_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when 12h ADX > 25 (trending market)
        # and 12h RSI is not extreme (avoid overextended entries)
        if adx_14_12h_aligned[i] > 25 and 30 < rsi_14_12h_aligned[i] < 70:
            # Long conditions:
            # 1. Price breaks above 6h Donchian upper channel
            # 2. Volume confirmation: volume > 1.5x average
            # 3. Price above Donchian midpoint (bullish bias)
            if (close[i] > highest_20[i] and 
                volume_ratio[i] > 1.5 and 
                close[i] > donchian_mid[i]):
                signals[i] = 0.25
                
            # Short conditions:
            # 1. Price breaks below 6h Donchian lower channel
            # 2. Volume confirmation: volume > 1.5x average
            # 3. Price below Donchian midpoint (bearish bias)
            elif (close[i] < lowest_20[i] and 
                  volume_ratio[i] > 1.5 and 
                  close[i] < donchian_mid[i]):
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ADX_RSI_Donchian_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0