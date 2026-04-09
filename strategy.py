#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and weekly trend filter
# - Uses 1d HTF for prior day's volume average (20-period) to confirm breakout strength
# - Uses 1w HTF for EMA(50) trend filter: only long when price > weekly EMA50, short when price < weekly EMA50
# - Long when price breaks above 20-period 6h Donchian high with volume > 1.5x 1d average volume
# - Short when price breaks below 20-period 6h Donchian low with same volume confirmation
# - Fixed position size 0.25 to control drawdown
# - Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# - Donchian channels provide clear breakout levels; volume confirms institutional participation
# - Weekly EMA filter avoids counter-trend trades in strong trends, works in both bull and bear markets

name = "6h_1d_1w_donchian_breakout_volume_trend_v1"
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
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    close_1w = df_1w['close'].values
    
    # Calculate prior day's 1d volume average (20-period) scaled for 6h comparison
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    # Scale daily average to 6h equivalent: daily volume / 4 (4x 6h bars in 1d)
    vol_ma_20_6h_equiv = vol_ma_20_1d_aligned / 4.0
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper channel = highest high of last 20 periods
    # Lower channel = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20_6h_equiv[i]) or np.isnan(ema_50_1w_aligned[i]) or
            vol_ma_20_6h_equiv[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x scaled 1d average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_6h_equiv[i]
        
        if position == 1:  # Long position
            # Exit long: price closes below Donchian lower channel
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short: price closes above Donchian upper channel
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation + weekly trend filter
            if volume_confirmed:
                # Long entry: price breaks above Donchian high AND price > weekly EMA50 (uptrend)
                if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND price < weekly EMA50 (downtrend)
                elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals