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
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h Donchian channels (15-period) for breakout signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_15 = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    donchian_low_15 = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_15)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_15)
    
    # Calculate 12h volume moving average for confirmation
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=15, min_periods=15).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
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
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid high volatility periods
        atr_threshold = np.nanpercentile(atr_14_1d_aligned[max(0, i-100):i+1], 70) if i >= 30 else atr_14_1d_aligned[i]
        low_volatility = atr_14_1d_aligned[i] < atr_threshold
        
        # Volume filter: current 12h volume above average
        volume_filter = vol_ma_12h_aligned[i] > 0 and volume[i] > vol_ma_12h_aligned[i] * 0.8
        
        # Breakout signals: price breaks 12h Donchian channels
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Long conditions: bullish trend + low volatility + volume + upward breakout
        long_condition = (price_above_ema and 
                         low_volatility and 
                         volume_filter and 
                         breakout_up)
        
        # Short conditions: bearish trend + low volatility + volume + downward breakout
        short_condition = (price_below_ema and 
                          low_volatility and 
                          volume_filter and 
                          breakout_down)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volatility spike
        elif position == 1 and (not price_above_ema or not low_volatility):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or not low_volatility):
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

name = "12h_DonchianBreakout_EMA34Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0