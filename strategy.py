#!/usr/bin/env python3
"""
4h_price_channel_breakout_1d_trend_volume_v1
Hypothesis: Price channel breakouts (Donchian 20) on 4h with 1d trend filter and volume confirmation work in both bull and bear markets.
In bull markets: buy breakouts above upper channel with 1d uptrend.
In bear markets: sell breakdowns below lower channel with 1d downtrend.
Uses strict volume confirmation and ATR filter to reduce false signals.
Target: 20-50 trades per year (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    # Use pandas rolling for proper min_periods handling
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter (optional, can be removed if too restrictive)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high + 1d uptrend + volume surge
        if (close[i] > donchian_high[i] and 
            close[i] > ema_1d_aligned[i] and  # 1d uptrend
            volume[i] > vol_ma[i] * 1.5):    # Volume confirmation
            signals[i] = 0.25
        
        # Short condition: price breaks below Donchian low + 1d downtrend + volume surge
        elif (close[i] < donchian_low[i] and 
              close[i] < ema_1d_aligned[i] and  # 1d downtrend
              volume[i] > vol_ma[i] * 1.5):     # Volume confirmation
            signals[i] = -0.25
        
        # Optional: exit when price crosses back to opposite side of channel
        # This helps prevent whipsaws but can be omitted for simplicity
        # elif position != 0 and ((position == 1 and close[i] < donchian_low[i]) or 
        #                        (position == -1 and close[i] > donchian_high[i])):
        #     signals[i] = 0.0
    
    return signals