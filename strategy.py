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
    
    # === 4h data for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h ATR for stop and trend
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4h EMA21 for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === 12h data for regime filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h ADX for trend strength
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_12h_smooth = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / tr_12h_smooth
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / tr_12h_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 4h Donchian breakout with volume confirmation ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume spike
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    warmup = 100
    
    # Track position and entry price for stop
    position = 0
    entry_price = 0.0
    
    for i in range(warmup, n):
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        price = close[i]
        atr_val = atr_4h_aligned[i]
        ema_val = ema_21_4h_aligned[i]
        adx_val = adx_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit conditions
        if position == 1:
            # Stop loss or trend reversal
            if price <= entry_price - 2.5 * atr_val or price < ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:
            # Stop loss or trend reversal
            if price >= entry_price + 2.5 * atr_val or price > ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic
        if position == 0 and in_session:
            # Long: price above Donchian upper, above EMA21, strong trend (ADX>25), volume spike
            if (price > highest_20[i] and price > ema_val and 
                adx_val > 25 and vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # Short: price below Donchian lower, below EMA21, strong trend (ADX>25), volume spike
            elif (price < lowest_20[i] and price < ema_val and 
                  adx_val > 25 and vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_EMA_ADX_Volume_Session"
timeframe = "4h"
leverage = 1.0