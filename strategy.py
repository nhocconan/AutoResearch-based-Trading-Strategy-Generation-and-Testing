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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily EMA(8) and EMA(21) for momentum
    ema_8_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_8_1d)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 4h Donchian channels (10-period) for breakout signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_10)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_10)
    
    # Calculate 4h volume moving average for confirmation
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=10, min_periods=10).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_8_1d_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Momentum filter: EMA8 > EMA21 for bullish, EMA8 < EMA21 for bearish
        ema_bullish = ema_8_1d_aligned[i] > ema_21_1d_aligned[i]
        ema_bearish = ema_8_1d_aligned[i] < ema_21_1d_aligned[i]
        
        # Volume filter: current 4h volume above average
        volume_filter = vol_ma_4h_aligned[i] > 0 and volume[i] > vol_ma_4h_aligned[i] * 1.2
        
        # Breakout signals: price breaks 4h Donchian channels
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Long conditions: bullish trend + momentum + volume + upward breakout
        long_condition = (price_above_ema and 
                         ema_bullish and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: bearish trend + momentum + volume + downward breakout
        short_condition = (price_below_ema and 
                          ema_bearish and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or momentum loss
        elif position == 1 and (not price_above_ema or not ema_bullish):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or not ema_bearish):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_EMA8_21_4hDonchianBreakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0