#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-week EMA trend filter and volume confirmation
# Works in bull markets by catching breakouts, works in bear markets by only taking shorts in downtrends
# Discrete position sizing (0.25) minimizes fee drag. Target: 20-50 trades/year.
# Uses 1d primary timeframe with 1h HTF for trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily indicators
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Daily 20-period EMA for trend filter
    daily_ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily Donchian channels (20-period)
    daily_highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    daily_lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume ratio (current vs 20-period average)
    daily_vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_volume_ratio = daily_volume / (daily_vol_ma_20 + 1e-10)
    
    # Align daily indicators to 1d timeframe (no additional delay needed for EMA/Donchian/volume)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, daily_ema_20)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, daily_highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, daily_lowest_20)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_ratio)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above daily EMA20 (uptrend filter)
        # 2. Price breaks above daily Donchian upper channel (breakout)
        # 3. Volume confirmation (1.5x average volume)
        # 4. Discrete position size: 0.25
        
        # Short conditions:
        # 1. Price below daily EMA20 (downtrend filter)
        # 2. Price breaks below daily Donchian lower channel (breakdown)
        # 3. Volume confirmation (1.5x average volume)
        # 4. Discrete position size: -0.25
        
        if (close[i] > ema_20_aligned[i] and  # Uptrend filter
            close[i] > highest_20_aligned[i] and     # Donchian breakout
            volume_ratio_aligned[i] > 1.5):        # Volume confirmation
            signals[i] = 0.25
            
        elif (close[i] < ema_20_aligned[i] and   # Downtrend filter
              close[i] < lowest_20_aligned[i] and      # Donchian breakdown
              volume_ratio_aligned[i] > 1.5):        # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_DailyEMA20_Donchian20_Volume_Breakout"
timeframe = "1d"
leverage = 1.0