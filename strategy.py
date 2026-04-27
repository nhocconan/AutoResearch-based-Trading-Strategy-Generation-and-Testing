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
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    # Get daily data for price action (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate ATR(14) for volatility filter
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
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(df_1d['close'].values, df_1w, ema_20_1w)
    # Align daily Donchian channels to daily timeframe (same timeframe, but using helper)
    donchian_high_aligned = align_htf_to_ltf(df_1d['close'].values, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(df_1d['close'].values, df_1d, donchian_low)
    
    # Align weekly EMA to 1d timeframe (already aligned above)
    # Now align everything to the actual price timeframe (1d)
    ema_20_1w_final = align_htf_to_ltf(prices, df_1d, ema_20_1w_aligned)
    donchian_high_final = align_htf_to_ltf(prices, df_1d, donchian_high_aligned)
    donchian_low_final = align_htf_to_ltf(prices, df_1d, donchian_low_aligned)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 20)  # Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_final[i]) or
            np.isnan(donchian_low_final[i]) or
            np.isnan(ema_20_1w_final[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # Trend filter: price above/below weekly EMA20
        uptrend = price > ema_20_1w_final[i]
        downtrend = price < ema_20_1w_final[i]
        
        if position == 0:
            # Long: break above Donchian high in uptrend with volume
            if volume_confirmation and uptrend and price > donchian_high_final[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in downtrend with volume
            elif volume_confirmation and downtrend and price < donchian_low_final[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to Donchian low or trend changes
            if price < donchian_low_final[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to Donchian high or trend changes
            if price > donchian_high_final[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0