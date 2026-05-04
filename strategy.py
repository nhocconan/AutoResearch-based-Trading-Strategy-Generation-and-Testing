#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian breakouts capture institutional participation; 12h EMA50 ensures trend alignment.
# Volume spike (2.0x 20-period EMA) filters weak breakouts. Designed for 6h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws.

name = "6h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels for each 6h bar using prior 12h bar's OHLC
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        # Need prior 12h bar data (12h bar must be closed)
        if i < 48:  # 48*15m = 12h, need at least one full 12h bar before current 6h bar
            continue
            
        # Get index of prior 12h bar in 12h dataframe
        # 12h bar index = floor(i / 48) - 1 (since we want prior completed 12h bar)
        idx_12h = (i // 48) - 1
        if idx_12h < 0 or idx_12h >= len(df_12h):
            continue
            
        # Calculate Donchian levels from prior 12h bar
        donchian_high[i] = df_12h['high'].iloc[idx_12h:idx_12h+20].max()
        donchian_low[i] = df_12h['low'].iloc[idx_12h:idx_12h+20].min()
    
    # Volume confirmation: 2.0x 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + price above 12h EMA50 (uptrend)
            if (close[i] > donchian_high[i] and volume_spike and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + price below 12h EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and volume_spike and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low OR price below 12h EMA50 (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian high OR price above 12h EMA50 (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals