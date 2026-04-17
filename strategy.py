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
    
    # Get weekly data for main trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter (slower = more reliable)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Donchian(20) for breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Use previous day's levels (avoid look-ahead)
    donchian_high_prev = np.roll(donchian_high, 1)
    donchian_low_prev = np.roll(donchian_low, 1)
    donchian_high_prev[0] = np.nan
    donchian_low_prev[0] = np.nan
    
    # Align daily Donchian levels to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high_prev)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low_prev)
    
    # Volume confirmation: current volume > 2.0 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need EMA34, Donchian, volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(donchian_high_1d[i]) or 
            np.isnan(donchian_low_1d[i]) or
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-day average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        # Weekly trend filter: price above/below weekly EMA34
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, volatility AND weekly uptrend
            if close[i] > donchian_high_1d[i] and volume_filter and volatility_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, volatility AND weekly downtrend
            elif close[i] < donchian_low_1d[i] and volume_filter and volatility_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly EMA34 or volatility drops
            if close[i] < ema34_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly EMA34 or volatility drops
            if close[i] > ema34_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA34_DonchianBreakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0