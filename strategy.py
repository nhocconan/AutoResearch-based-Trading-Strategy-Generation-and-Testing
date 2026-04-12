#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_Trend_v1
Hypothesis: Trade weekly Donchian channel breakouts (20-week high/low) on daily timeframe with trend confirmation from weekly EMA50 and volume filter. 
Long when price breaks above weekly Donchian high with volume > 1.5x average and price above weekly EMA50. 
Short when price breaks below weekly Donchian low with volume > 1.5x average and price below weekly EMA50.
Exit when price crosses weekly EMA50 in opposite direction. 
Designed for 15-25 trades/year with clear trend-following logic that works in bull (breakouts continue) and bear (breakouts fail, reverse) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR DONCHIAN CHANNEL AND EMA50 ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high_1w).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low_1w).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly average volume for volume filter
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === DAILY INDICATORS ===
    # Daily average volume for volume confirmation
    vol_ma_daily = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(vol_ma_daily[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength: current daily volume > 1.5x weekly average volume
        strong_volume = volume[i] > (vol_ma_1w_aligned[i] * 1.5)
        
        # Long: price breaks above weekly Donchian high with volume and above weekly EMA50
        long_signal = (close[i] > donchian_high_aligned[i] and 
                      strong_volume and 
                      close[i] > ema50_1w_aligned[i])
        
        # Short: price breaks below weekly Donchian low with volume and below weekly EMA50
        short_signal = (close[i] < donchian_low_aligned[i] and 
                       strong_volume and 
                       close[i] < ema50_1w_aligned[i])
        
        # Exit: price crosses weekly EMA50 in opposite direction
        exit_long = (position == 1 and close[i] < ema50_1w_aligned[i])
        exit_short = (position == -1 and close[i] > ema50_1w_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals