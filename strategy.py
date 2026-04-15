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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 1w HTF data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for higher timeframe trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 4h Donchian channels (20-period)
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: UTC 8-20 (most active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(ema_200_4h[i]) or np.isnan(ema_50_4h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above Donchian upper (20) - bullish breakout
        # 2. Price above 1d EMA200 (bullish higher timeframe trend)
        # 3. Price above 1w EMA50 (bullish higher timeframe trend)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        if (close[i] > upper_20[i] and
            close[i] > ema_200_4h[i] and
            close[i] > ema_50_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below Donchian lower (20) - bearish breakdown
        # 2. Price below 1d EMA200 (bearish higher timeframe trend)
        # 3. Price below 1w EMA50 (bearish higher timeframe trend)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price
        elif (close[i] < lower_20[i] and
              close[i] < ema_200_4h[i] and
              close[i] < ema_50_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_EMA200_1w_EMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0