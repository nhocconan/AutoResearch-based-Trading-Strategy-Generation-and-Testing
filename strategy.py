#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w volume confirmation and 1d trend filter
    # Designed for low trade frequency (12-37/year) to minimize fee drag
    # Works in bull/bear markets by following institutional breakouts with volume confirmation
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 1w data for HTF volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.ones(len(df_1w))
    
    # Calculate 1w volume average (10-period)
    vol_avg_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_10_1w)
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_avg_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.8x 10-period average
        volume_confirmed = volume_1w[i // (7*24*4)] > 1.8 * vol_avg_10_1w_aligned[i]
        
        # Trend filter: only trade in direction of 1d EMA200
        trend_filter_long = close[i] > ema200_1d_aligned[i]
        trend_filter_short = close[i] < ema200_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_20[i-1]  # Break above upper channel
        breakout_short = close[i] < low_20[i-1]  # Break below lower channel
        
        # Entry conditions
        enter_long = breakout_long and volume_confirmed and trend_filter_long
        enter_short = breakout_short and volume_confirmed and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout or midpoint reversion
        donchian_mid = (high_20[i] + low_20[i]) / 2
        exit_long = position == 1 and (close[i] < donchian_mid or close[i] < low_20[i])
        exit_short = position == -1 and (close[i] > donchian_mid or close[i] > high_20[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0