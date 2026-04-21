#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h ATR for volatility and stop
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Donchian breakout levels (20-period)
    high_4h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_4h = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_12h = ema_50_12h_aligned[i]
        atr_12h = atr_14_12h[i]
        vol_ratio_val = vol_ratio[i]
        upper_channel = high_4h[i]
        lower_channel = low_4h[i]
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian upper, above 12h EMA, volume spike
            if (price_close > upper_channel and 
                price_close > ema_12h and 
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 4h Donchian lower, below 12h EMA, volume spike
            elif (price_close < lower_channel and 
                  price_close < ema_12h and 
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility spike (ATR-based stop)
            if position == 1:
                if (price_close < ema_12h or 
                    price_low < (ema_12h - 2.0 * atr_12h) or  # stop loss
                    price_high > upper_channel + 0.5 * atr_12h):  # take profit near channel
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (price_close > ema_12h or 
                    price_high > (ema_12h + 2.0 * atr_12h) or  # stop loss
                    price_low < lower_channel - 0.5 * atr_12h):  # take profit near channel
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0