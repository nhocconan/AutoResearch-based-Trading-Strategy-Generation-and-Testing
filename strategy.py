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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    donchian_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h ATR(14) for volatility filter (primary TF)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 6h volume moving average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 6h ATR is elevated (> 0.4% of price)
        vol_filter = atr_14[i] > 0.004 * close[i]
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_filter = volume[i] > 1.2 * vol_ma_20[i]
        
        # Long conditions:
        # 1. Price above weekly EMA34 (bullish bias)
        # 2. Price breaks above weekly Donchian(20) high
        # 3. Volatility filter
        # 4. Volume confirmation
        if (close[i] > ema_34_1w_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            vol_filter and volume_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA34 (bearish bias)
        # 2. Price breaks below weekly Donchian(20) low
        # 3. Volatility filter
        # 4. Volume confirmation
        elif (close[i] < ema_34_1w_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              vol_filter and volume_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian20_EMA34_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0