#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on daily close using pandas for correct min_periods
    close_1d_series = pd.Series(df_1d['close'])
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 1h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 14-period ATR
    high_low = high - low
    high_close = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr = np.concatenate([[np.nan], tr[1:]])  # First element is NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period high/low for Donchian breakout
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(34, 14, 20, 20) + 5
    
    # Pre-compute session hours for filtering (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and price above daily EMA34
            if price > high_max[i] and vol_ratio > 2.0 and price > ema_34_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and price below daily EMA34
            elif price < low_min[i] and vol_ratio > 2.0 and price < ema_34_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian20_EMA34_Trend_Volume_ATRStop_Session_v1"
timeframe = "1h"
leverage = 1.0