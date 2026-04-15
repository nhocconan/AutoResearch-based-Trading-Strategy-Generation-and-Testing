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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) and EMA(89) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(df_1d['close'].values).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align to 6h
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_1d_6h = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_20 + lowest_20) / 2.0
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_6h[i]) or np.isnan(ema_89_1d_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA34 > EMA89 for bullish bias, EMA34 < EMA89 for bearish bias
        bullish_trend = ema_34_1d_6h[i] > ema_89_1d_6h[i]
        bearish_trend = ema_34_1d_6h[i] < ema_89_1d_6h[i]
        
        # Volatility filter: ATR > 0.5% of price ensures sufficient momentum
        vol_filter = atr_14_6h[i] > 0.005 * close[i]
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume_ratio[i] > 1.5
        
        # Long conditions:
        # 1. Bullish daily trend (EMA34 > EMA89)
        # 2. Price breaks above 20-period Donchian high with volume
        # 3. Volatility and volume filters
        if (bullish_trend and
            close[i] > highest_20[i] and
            vol_filter and
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Bearish daily trend (EMA34 < EMA89)
        # 2. Price breaks below 20-period Donchian low with volume
        # 3. Volatility and volume filters
        elif (bearish_trend and
              close[i] < lowest_20[i] and
              vol_filter and
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA3489_Trend_Donchian20_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0