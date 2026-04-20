#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Donchian_Trend_10_20_Volume_15"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hour-based session filter (UTC 8-20) computed once
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h: 10-period EMA for short-term trend ===
    close_4h = df_4h['close'].values
    ema_10_4h = pd.Series(close_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_10_4h)
    
    # === 4h: 20-period EMA for medium-term trend ===
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 1d: Donchian channels (20-day high/low) using previous day's data ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # 20-day Donchian channels (highest high, lowest low over 20 days)
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(prev_high[i-20:i])
        donchian_low[i] = np.min(prev_low[i-20:i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Main loop ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_10_val = ema_10_4h_aligned[i]
        ema_20_val = ema_20_4h_aligned[i]
        upper_donchian = donchian_high_aligned[i]
        lower_donchian = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_10_val) or np.isnan(ema_20_val) or 
            np.isnan(upper_donchian) or np.isnan(lower_donchian) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA10 > EMA20 (uptrend), price above upper Donchian, volume confirmation
            if (ema_10_val > ema_20_val and  # Uptrend filter
                close_val > upper_donchian and   # Break above Donchian high
                vol_ratio_val > 1.5):        # Volume confirmation
                signals[i] = 0.15
                position = 1
            # Short: EMA10 < EMA20 (downtrend), price below lower Donchian, volume confirmation
            elif (ema_10_val < ema_20_val and  # Downtrend filter
                  close_val < lower_donchian and   # Break below Donchian low
                  vol_ratio_val > 1.5):        # Volume confirmation
                signals[i] = -0.15
                position = -1
        
        elif position == 1:
            # Long exit: EMA10 < EMA20 (trend change) or price breaks below lower Donchian
            if ema_10_val < ema_20_val or close_val < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.15
        
        elif position == -1:
            # Short exit: EMA10 > EMA20 (trend change) or price breaks above upper Donchian
            if ema_10_val > ema_20_val or close_val > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.15
    
    return signals