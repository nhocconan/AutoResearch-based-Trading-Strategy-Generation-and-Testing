#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyEMA34_Trend_VolumeConfirm
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume confirmation captures sustained moves in both bull and bear markets while avoiding whipsaws. Weekly EMA ensures alignment with higher timeframe momentum, volume confirmation filters low-conviction breakouts, and Donchian channels provide objective breakout levels. Designed for low trade frequency (~15-30/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Daily Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper and lower Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly EMA34 trend filter ===
    weekly_close = df_1w['close'].values
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume confirmation (20-period on daily) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_confirm = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + weekly uptrend + volume confirmation
            if price_close > upper_channel[i] and price_close > weekly_trend and vol_confirm > 1.3:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below lower Donchian + weekly downtrend + volume confirmation
            elif price_close < lower_channel[i] and price_close < weekly_trend and vol_confirm > 1.3:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Exit: price reverts to opposite Donchian channel or weekly trend changes
            if position == 1:
                if price_close < lower_channel[i] or price_close < weekly_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > upper_channel[i] or price_close > weekly_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0