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
    
    # Get 12h data for trend and structure
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian channels (20-period)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 4h
    upper_12h = align_htf_to_ltf(prices, df_12h, high_20_12h)
    lower_12h = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 4h volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h[i]) or 
            np.isnan(lower_12h[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price relative to 12h EMA34
        price_above_ema = close[i] > ema34_12h_aligned[i]
        price_below_ema = close[i] < ema34_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_12h[i]
        breakout_down = close[i] < lower_12h[i]
        
        if position == 0:
            # Long: 12h Donchian breakout up + volume + above 12h EMA34
            if (breakout_up and volume_filter and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: 12h Donchian breakout down + volume + below 12h EMA34
            elif (breakout_down and volume_filter and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: stop loss or reversal
            # Stop loss: 2 * ATR below entry (approximated by 12h EMA or lower band)
            if (close[i] < ema34_12h_aligned[i] - 2.0 * atr[i]) or \
               (close[i] < lower_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stop loss or reversal
            # Stop loss: 2 * ATR above entry
            if (close[i] > ema34_12h_aligned[i] + 2.0 * atr[i]) or \
               (close[i] > upper_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0