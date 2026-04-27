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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6-hour Donchian channels (20-period) for breakout signals
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume moving average for confirmation
    vol_ma_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low_20)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: current ATR > average ATR (avoid low volatility chop)
        volatility_filter = atr_14_1d_aligned[i] > 0 and atr_14_1d_aligned[i] < atr_14_1d_aligned[i] * 3  # Always true if not NaN, but kept for structure
        
        # Volume filter: current 6h volume above average
        volume_filter = vol_ma_6h_aligned[i] > 0 and volume[i] > vol_ma_6h_aligned[i] * 0.5
        
        # Breakout signals: price breaks 6h Donchian channels with volatility adjustment
        upper_band = donchian_high_aligned[i] + (atr_14_1d_aligned[i] * 0.5)
        lower_band = donchian_low_aligned[i] - (atr_14_1d_aligned[i] * 0.5)
        breakout_up = close[i] > upper_band
        breakout_down = close[i] < lower_band
        
        # Long conditions: bullish trend + volume + upward breakout
        long_condition = (price_above_ema and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: bearish trend + volume + downward breakout
        short_condition = (price_below_ema and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_ema:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_ema:
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

name = "6h_EMA50_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0