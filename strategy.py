#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# - Long when price breaks above 20-bar high AND close > 1d EMA50 AND volume > 1.5x 20-bar avg
# - Short when price breaks below 20-bar low AND close < 1d EMA50 AND volume > 1.5x 20-bar avg
# - Exit on opposite Donchian breakout or when trend filter fails
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Targets ~30 trades/year (120 total over 4 years) to avoid fee drag
# - 1d trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation reduces false breakouts
# - Works in both bull (trend continuation) and bear (trend reversal) markets

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above 20-bar high, 1d uptrend, volume spike
            if (close[i] > highest_high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.30
            # Short signal: price breaks below 20-bar low, 1d downtrend, volume spike
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit long when price breaks below 20-bar low or trend fails
            if position == 1 and (close[i] < lowest_low_20[i] or close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Exit short when price breaks above 20-bar high or trend fails
            elif position == -1 and (close[i] > highest_high_20[i] or close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals