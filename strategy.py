#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h/1d Donchian breakout + volume confirmation + ATR filter.
Long when price breaks above 4h Donchian(10) high with volume > 1.3x 20-period average and ATR(14) > 0.
Short when price breaks below 4h Donchian(10) low with volume > 1.3x 20-period average and ATR(14) > 0.
Use discrete position sizing of 0.20 to limit fee drag and manage drawdown.
Target: 60-150 total trades over 4 years (15-37/year) to avoid overtrading.
Use 4h/1d for SIGNAL DIRECTION, 1h only for ENTRY TIMING.
Add session filter (08-20 UTC) to reduce noise trades.
"""

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
    
    # Get 4h data for Donchian and ATR
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian Channel (10)
    period10_high = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    period10_low = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    donchian_high = period10_high
    donchian_low = period10_low
    
    # Calculate 4h ATR (14)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(vol_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = vol_4h_aligned[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Trend filter: price above/below 1d EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and trend
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirmed and 
                price_above_ema and
                atr_aligned[i] > 0):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low with volume and trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirmed and 
                  price_below_ema and
                  atr_aligned[i] > 0):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Donchian low or trend reverses
            if (close[i] < donchian_low_aligned[i] or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises above Donchian high or trend reverses
            if (close[i] > donchian_high_aligned[i] or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian10_Volume_EMA50_Session"
timeframe = "1h"
leverage = 1.0