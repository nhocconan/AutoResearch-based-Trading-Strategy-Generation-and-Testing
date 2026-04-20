#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for 20-day Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels on daily data
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Calculate 12h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4x ATR stop level
    atr_stop = 4 * atr_12h
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 12h volume > 15-period average
    volume = prices['volume'].values
    volume_ma_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    # Calculate 12h RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in HTF indicators or ATR
        if np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or np.isnan(atr_12h[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_15[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi_values[i] > 30) and (rsi_values[i] < 70)
        
        # Price levels
        upper_band = highest_20d_aligned[i]
        lower_band = lowest_20d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with volume and RSI filter
            if price > upper_band and vol_filter and rsi_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakdown: price breaks below 20-day low with volume and RSI filter
            elif price < lower_band and vol_filter and rsi_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (4x ATR below entry) or price breaks below 20-day low
            if price <= entry_price - atr_stop[i] or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (4x ATR above entry) or price breaks above 20-day high
            if price >= entry_price + atr_stop[i] or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_Volume_RSI_Filter_Session_ATRStop"
timeframe = "12h"
leverage = 1.0