# 1d Weekly Donchian Breakout with Volume Spike and ATR Filter
# Targets breakouts of weekly highs/lows with volume confirmation and volatility filter
# Designed to work in both bull (breakout continuation) and bear (breakdown continuation) markets
# Weekly timeframe reduces trade frequency to avoid fee drag, daily allows timely entry

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_Volume_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-week lookback)
    # Using rolling window on weekly data
    weekly_high_max = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_low_min = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (waits for weekly bar to close)
    weekly_high_max_daily = align_htf_to_ltf(prices, df_weekly, weekly_high_max)
    weekly_low_min_daily = align_htf_to_ltf(prices, df_weekly, weekly_low_min)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter and stop calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_max_daily[i]) or np.isnan(weekly_low_min_daily[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol > 1.5 * vol_ma
        
        # Volatility filter: ATR > 50-day average ATR (avoid low volatility choppy periods)
        atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = atr_val > 0.5 * atr_ma_50 if not np.isnan(atr_ma_50) else True
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume and vol filter
            if price > weekly_high_max_daily[i] and volume_spike and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with volume and vol filter
            elif price < weekly_low_min_daily[i] and volume_spike and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly Donchian low (reversal signal) or ATR-based stop
            if price < weekly_low_min_daily[i] or price < high[max(0, i-1)] - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly Donchian high (reversal signal) or ATR-based stop
            if price > weekly_high_max_daily[i] or price > low[max(0, i-1)] + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals