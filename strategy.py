#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_breakout_v2
# Hypothesis: 6h strategy using weekly Donchian(20) breakout in direction of weekly Camarilla pivot bias.
# Long: Price breaks above weekly Donchian high, weekly close > weekly pivot (PP), and volume > 1.5x 20-period average.
# Short: Price breaks below weekly Donchian low, weekly close < weekly pivot (PP), and volume > 1.5x 20-period average.
# Exit: Opposite Donchian break or ATR trailing stop (2.0x ATR from extreme).
# Uses weekly structure for bias and breakout levels, volume to filter weak moves, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for weekly structure (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian(20) channels
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly Camarilla pivot point (PP = (H+L+C)/3)
    camarilla_pp = (high_1w + low_1w + close_1w) / 3.0
    
    # Align HTF weekly data to 6h timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Weekly bias: close above/below pivot
        weekly_bullish = close_1w[min(len(close_1w)-1, i//20)] > camarilla_pp[min(len(camarilla_pp)-1, i//20)] if i >= 20 else False
        weekly_bearish = close_1w[min(len(close_1w)-1, i//20)] < camarilla_pp[min(len(camarilla_pp)-1, i//20)] if i >= 20 else False
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below weekly Donchian low
            elif low[i] < donchian_low_aligned[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price breaks above weekly Donchian high
            elif high[i] > donchian_high_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly Donchian high, weekly bullish bias, volume confirmed
            if (high[i] > donchian_high_aligned[i] and weekly_bullish and volume_confirmed):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below weekly Donchian low, weekly bearish bias, volume confirmed
            elif (low[i] < donchian_low_aligned[i] and weekly_bearish and volume_confirmed):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals