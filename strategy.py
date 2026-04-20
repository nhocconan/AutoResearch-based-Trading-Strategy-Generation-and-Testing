#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1h HTF data once for trend and volatility
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 5:
        return np.zeros(n)
    
    # Calculate 1h EMA200 for trend
    close_1h = df_1h['close'].values
    ema_200 = pd.Series(close_1h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1h ATR for volatility filter
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h_prev = np.roll(close_1h, 1)
    close_1h_prev[0] = close_1h[0]
    tr1 = np.abs(high_1h - low_1h)
    tr2 = np.abs(high_1h - close_1h_prev)
    tr3 = np.abs(low_1h - close_1h_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1h indicators to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1h, ema_200)
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_200_aligned[i]) or np.isnan(atr_1h_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_200_aligned[i]
        atr_val = atr_1h_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        vol_ok = vol_filter[i]
        
        # Volatility filter: only trade when ATR > 0
        vol_filter_ok = atr_val > 0
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume
            if price > upper and price > ema_val and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume
            elif price < lower and price < ema_val and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend changes
            if price < lower or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend changes
            if price > upper or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1h_DonchianEMA200_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0