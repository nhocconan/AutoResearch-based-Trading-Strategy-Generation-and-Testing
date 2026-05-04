#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA34 ensures trend alignment to avoid counter-trend trades.
# Volume confirmation (2.0x 20-period EMA) filters weak breakouts. Designed for 12h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.30).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws.

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    # We need to calculate this on 12h data then align to 12h prices
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper and lower bands on 12h data
    donchian_upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h price index
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # Volume confirmation: 2.0x 20-period EMA on 12h volume
    vol_12h = df_12h['volume'].values
    vol_ema_20_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = prices['volume'].iloc[i] > (2.0 * vol_ema_20_12h_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + price above 1d EMA34 (uptrend)
            if (close[i] > donchian_upper_aligned[i] and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower + volume spike + price below 1d EMA34 (downtrend)
            elif (close[i] < donchian_lower_aligned[i] and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian lower OR price below 1d EMA34 (trend change)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Donchian upper OR price above 1d EMA34 (trend change)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals