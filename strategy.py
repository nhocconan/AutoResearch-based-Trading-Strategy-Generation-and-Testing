#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Squeeze_With_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for Keltner channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h ATR(20) for Keltner channels
    tr4h = np.maximum(df_4h['high'].values - df_4h['low'].values,
                      np.maximum(np.abs(df_4h['high'].values - np.concatenate([[np.nan], df_4h['close'][:-1]])),
                                 np.abs(df_4h['low'].values - np.concatenate([[np.nan], df_4h['close'][:-1]]))))
    atr20_4h = pd.Series(tr4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA20 for Keltner center
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner bands
    upper_keltner = ema20_4h + 2.0 * atr20_4h
    lower_keltner = ema20_4h - 2.0 * atr20_4h
    
    # Calculate Bollinger Bands for squeeze detection (20,2)
    sma20_4h = pd.Series(df_4h['close']).rolling(window=20, min_periods=20).mean().values
    std20_4h = pd.Series(df_4h['close']).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_4h + 2.0 * std20_4h
    lower_bb = sma20_4h - 2.0 * std20_4h
    
    # Squeeze condition: BB inside Keltner
    squeeze = (upper_bb <= upper_keltner) & (lower_bb >= lower_keltner)
    
    # Volume filter: current 1d volume > 1.5 * 20-period average
    vol_series_1d = pd.Series(df_1d['volume'].values)
    vol_ma_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma_1d * 1.5)
    
    # Align all to 4h
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    squeeze_4h = align_htf_to_ltf(prices, df_4h, squeeze)
    volume_filter_1d_4h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volatility measures
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_4h[i]) or np.isnan(squeeze_4h[i]) or 
            np.isnan(volume_filter_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema34_1d_4h[i]
        is_squeeze = squeeze_4h[i]
        vol_filter = volume_filter_1d_4h[i]
        
        if position == 0:
            # Enter long: squeeze breakout above upper Keltner with volume and uptrend
            if close[i] > upper_keltner[i] and close[i] > trend and vol_filter and is_squeeze:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breakout below lower Keltner with volume and downtrend
            elif close[i] < lower_keltner[i] and close[i] < trend and vol_filter and is_squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA20 (mean reversion) or squeeze fires in opposite direction
            if close[i] < ema20_4h[i] or (not is_squeeze and close[i] < ema20_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above EMA20 (mean reversion) or squeeze fires in opposite direction
            if close[i] > ema20_4h[i] or (not is_squeeze and close[i] > ema20_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals