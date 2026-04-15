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
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 12h (no alignment needed for same TF, but keep for consistency)
    upper_20_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_20_12h)
    lower_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_20_12h)
    
    # Get 1w HTF data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for weekly trend
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: focus on active UTC hours (8:00-20:00) for better liquidity
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_12h_aligned[i]) or np.isnan(lower_20_12h_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h price breaks above 12h Donchian upper (20) - bullish breakout
        # 2. Weekly EMA(21) bullish: price above weekly EMA
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_12h_aligned[i] and
            close[i] > ema_21_1w_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below 12h Donchian lower (20) - bearish breakdown
        # 2. Weekly EMA(21) bearish: price below weekly EMA
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_12h_aligned[i] and
              close[i] < ema_21_1w_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_12h_Donchian20_1w_EMA21_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0