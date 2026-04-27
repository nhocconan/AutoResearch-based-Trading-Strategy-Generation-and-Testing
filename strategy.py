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
    
    # Daily ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_daily = np.full(n, np.nan)
    for i in range(14, n):
        atr_daily[i] = np.mean(tr[i-14:i])
    
    # Daily EMA(34) for trend filter
    close_series = pd.Series(close)
    ema_34_daily = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily Donchian(20) channels
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        if (np.isnan(atr_daily[i]) or 
            np.isnan(ema_34_daily[i]) or
            np.isnan(donch_high_20[i]) or
            np.isnan(donch_low_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_daily[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume (tight to reduce trades)
        volume_confirmation = vol_ratio > 2.0
        
        # Trend filter: price above/below EMA34
        uptrend = price > ema_34_daily[i]
        downtrend = price < ema_34_daily[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if (volume_confirmation and 
                price > donch_high_20[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif (volume_confirmation and 
                  price < donch_low_20[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA34 or ATR-based stop
            if (price < ema_34_daily[i] or 
                price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above EMA34 or ATR-based stop
            if (price > ema_34_daily[i] or 
                price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_Donchian20_EMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0