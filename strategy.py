#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h Donchian trend filter and volume confirmation.
# Long when 1h EMA20 crosses above EMA50 AND price > 4h Donchian upper channel AND volume > 1.5x 20-bar avg.
# Short when 1h EMA20 crosses below EMA50 AND price < 4h Donchian lower channel AND volume > 1.5x 20-bar avg.
# Uses 1h timeframe with strict filters to target 15-30 trades/year.
# 4h Donchian provides medium-term trend structure to avoid counter-trend whipsaws.
# Volume confirmation reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity hours.
# Discrete sizing 0.20 minimizes fee churn while maintaining sufficient exposure.

name = "1h_EMA20_50_4hDonchian_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    dh_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_4h, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_4h, dl_20)
    
    # 1h EMA20 and EMA50 for crossover signals
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA crossover signals
    ema_cross_up = (ema_20 > ema_50) & (np.roll(ema_20, 1) <= np.roll(ema_50, 1))
    ema_cross_down = (ema_20 < ema_50) & (np.roll(ema_20, 1) >= np.roll(ema_50, 1))
    
    # Volume confirmation: current volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Donchian calculations
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: EMA20 crosses above EMA50 AND price > 4h Donchian upper AND volume confirmation
            if (ema_cross_up[i] and 
                curr_close > dh_20_aligned[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: EMA20 crosses below EMA50 AND price < 4h Donchian lower AND volume confirmation
            elif (ema_cross_down[i] and 
                  curr_close < dl_20_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: EMA20 crosses below EMA50 OR price < 4h Donchian lower (trend change)
            if (ema_cross_down[i] or 
                curr_low < dl_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: EMA20 crosses above EMA50 OR price > 4h Donchian upper (trend change)
            if (ema_cross_up[i] or 
                curr_high > dh_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals