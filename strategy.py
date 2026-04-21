#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend and structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Donchian(20) channels for breakout signals
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 4h ATR(14) for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_4h, atr_ratio)
    
    # Volume confirmation: volume / 30-period average volume (4h)
    vol_ma_30 = pd.Series(df_4h['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio_4h = df_4h['volume'].values / vol_ma_30
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_4h_aligned[i]
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, uptrend, volume spike, moderate volatility
            if (price_close > upper_band and 
                price_close > ema_trend and 
                vol_ratio > 1.5 and 
                atr_ratio_val > 0.7 and atr_ratio_val < 2.0):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below Donchian low, downtrend, volume spike, moderate volatility
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  vol_ratio > 1.5 and 
                  atr_ratio_val > 0.7 and atr_ratio_val < 2.0):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_close < lower_band or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_band or atr_ratio_val > 2.5 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_4hDonchianBreakout_4hTrend_VolumeATR_Session"
timeframe = "1h"
leverage = 1.0