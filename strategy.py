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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema34 = np.full(len(weekly_close), np.nan)
    for i in range(len(weekly_close)):
        if i < 34:
            weekly_ema34[i] = np.mean(weekly_close[:i+1])
        else:
            weekly_ema34[i] = weekly_ema34[i-1] * 0.94117647 + weekly_close[i] * 0.05882353  # EMA 34 alpha
    
    # Align weekly EMA34 to daily
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Get daily data for price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily high/low for Donchian breakout
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Donchian(20) channels
    donchian_high = np.full(len(daily_high), np.nan)
    donchian_low = np.full(len(daily_low), np.nan)
    for i in range(len(daily_high)):
        if i >= 20:
            donchian_high[i] = np.max(daily_high[i-20:i])
            donchian_low[i] = np.min(daily_low[i-20:i])
        else:
            donchian_high[i] = np.max(daily_high[:i+1])
            donchian_low[i] = np.min(daily_low[:i+1])
    
    # Align Donchian channels to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Daily volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Start after all indicators are ready
    start_idx = max(20, 20)  # Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_ema34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # Volatility filter: ATR above 50% of its 20-period average
        if i >= 20:
            atr_avg = np.mean(atr[max(0, i-20):i+1])
            vol_filter = atr[i] > atr_avg * 0.5
        else:
            vol_filter = True
        
        # Trend filter: price above/below weekly EMA34
        uptrend = price > weekly_ema34_aligned[i]
        downtrend = price < weekly_ema34_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume, volatility, and uptrend
            if volume_confirmation and vol_filter and uptrend and price > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume, volatility, and downtrend
            elif volume_confirmation and vol_filter and downtrend and price < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below Donchian low or trend changes
            if price < donchian_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high or trend changes
            if price > donchian_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA34Trend_Volume"
timeframe = "1d"
leverage = 1.0