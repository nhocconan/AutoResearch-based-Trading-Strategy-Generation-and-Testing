#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot with volume spike and 1d/1w trend filter
# Long when price closes above Camarilla H3 + volume > 2x average + 1d uptrend
# Short when price closes below Camarilla L3 + volume > 2x average + 1d downtrend
# Exit when price returns to Camarilla pivot or trend reverses
# Designed for 15-30 trades/year on 12h timeframe with strong trend capture and low turnover

name = "12h_1d_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day (using daily data)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    for i in range(n):
        # Find the most recent completed daily bar
        daily_idx = i // 2  # 2x 12h bars per day
        if daily_idx >= 1 and daily_idx < len(df_1d):
            # Previous day's OHLC (since current day is still forming)
            prev_daily_idx = daily_idx - 1
            if prev_daily_idx >= 0:
                ph = df_1d['high'].iloc[prev_daily_idx]
                pl = df_1d['low'].iloc[prev_daily_idx]
                pc = df_1d['close'].iloc[prev_daily_idx]
                camarilla_pivot[i] = (ph + pl + pc) / 3
                camarilla_h3[i] = camarilla_pivot[i] + (ph - pl) * 1.1 / 6
                camarilla_l3[i] = camarilla_pivot[i] - (ph - pl) * 1.1 / 6
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filter: both 1d and 1w EMAs agree
        is_uptrend = (close[i] > ema_50_1d_aligned[i]) and (close[i] > ema_50_1w_aligned[i])
        is_downtrend = (close[i] < ema_50_1d_aligned[i]) and (close[i] < ema_50_1w_aligned[i])
        
        # Entry conditions
        long_entry = (close[i] > camarilla_h3[i]) and volume_filter and is_uptrend
        short_entry = (close[i] < camarilla_l3[i]) and volume_filter and is_downtrend
        
        # Exit conditions
        long_exit = (close[i] < camarilla_pivot[i]) or (not is_uptrend)
        short_exit = (close[i] > camarilla_pivot[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals