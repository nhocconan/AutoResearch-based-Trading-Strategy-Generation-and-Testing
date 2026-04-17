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
    
    # Get daily data for 1d EMA34 and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume SMA20 for volume spike filter
    volume_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma20_1d)
    
    # Get weekly data for ADX (trend strength filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX (14-period)
    period = 14
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Align weekly ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 12h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA34, ADX, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_sma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Filters
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        strong_trend = adx_aligned[i] > 25
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        daily_volume_spike = volume_1d[i // 288] > (1.5 * volume_sma20_1d_aligned[i]) if i >= 288 else False
        
        if position == 0:
            # Long: price above EMA34, strong uptrend, volume confirmation
            if (price_above_ema and strong_trend and volume_filter and daily_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA34, strong downtrend, volume confirmation
            elif ((not price_above_ema) and strong_trend and volume_filter and daily_volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34 or trend weakens
            if (not price_above_ema) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 or trend weakens
            if price_above_ema or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_ADX_Volume_Spike_Filter"
timeframe = "12h"
leverage = 1.0