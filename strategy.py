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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = high
    low_4h = low
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h_calc = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: UTC 8-20 (most active hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(atr_14_4h_calc[i]) or np.isnan(volume_ratio[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. Price above 1d EMA(50) (bullish bias from daily trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if (close[i] > upper_20_4h[i] and
            close[i] > ema_50_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_4h_calc[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. Price below 1d EMA(50) (bearish bias from daily trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < lower_20_4h[i] and
              close[i] < ema_50_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_4h_calc[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_EMA50_Donchian20_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0