#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA34) and volume confirmation
    # Trade only in direction of 1w EMA34 to avoid counter-trend whipsaws
    # Volume spike (>2.0x 20-period average) confirms participation
    # Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 1w trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous 1d bar's Donchian channels (20-period)
    # Upper = max(high_prev_20), Lower = min(low_prev_20)
    upper_20 = np.full(len(df_1d), np.nan)
    lower_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        upper_20[i] = np.max(high_1d[i-20:i])
        lower_20[i] = np.min(low_1d[i-20:i])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Get 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1w volume for confirmation (>2.0x 20-period average)
    vol_ma_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    volume_spike_1w = volume_1w > (2.0 * vol_ma_1w)
    
    # Align all indicators to LTF (1d)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_20_aligned[i]
        short_breakout = close[i] < lower_20_aligned[i]
        
        # 1w trend filter
        bullish_trend = close[i] > ema34_1w_aligned[i]
        bearish_trend = close[i] < ema34_1w_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or not bullish_trend
        short_exit = long_breakout or not bearish_trend
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_ema34_volume_v1"
timeframe = "1d"
leverage = 1.0