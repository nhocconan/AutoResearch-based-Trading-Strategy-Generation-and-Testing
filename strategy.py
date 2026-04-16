#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for trend and volume) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1h data (for entry timing) ===
    # Calculate 1h ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = np.nan
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d EMA34 (trend filter) ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (for volume filter) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1h volume ratio (entry confirmation) ===
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = volume / vol_ma_20_1h
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_1h[i]) or 
            np.isnan(vol_ratio_1h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        vol_ratio = vol_ratio_1h[i]
        in_sess = in_session[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA OR volume dries up
            if price < ema_trend or volume < vol_ma_1d * 0.5:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above EMA OR volume dries up
            if price > ema_trend or volume < vol_ma_1d * 0.5:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat and in session) ===
        if position == 0 and in_sess:
            # LONG: Price above EMA with above-average volume
            if price > ema_trend and vol_ratio > 1.3:
                signals[i] = 0.20
                position = 1
                continue
            # SHORT: Price below EMA with above-average volume
            elif price < ema_trend and vol_ratio > 1.3:
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA34_VolumeBreak_Session"
timeframe = "1h"
leverage = 1.0