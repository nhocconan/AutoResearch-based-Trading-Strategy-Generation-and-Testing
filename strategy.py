#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v2
# Hypothesis: 12h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP(14) > 61.8 for mean reversion, < 38.2 for trend following).
# In choppy markets, trade mean reversion at Donchian bands; in trending markets, trade breakouts.
# Uses daily and weekly HTF for regime context: only trade if price is above/below weekly EMA200.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 12-30 trades/year.
# Uses MTF data loaded ONCE before loop as required.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily HTF data ONCE for weekly EMA200 context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Weekly EMA200 from daily data (proxy for weekly trend)
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Load 1d HTF data for chop calculation (using daily candles)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d_arr[0])
    tr3[0] = np.abs(low_1d[0] - close_1d_arr[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Chopiness Index: CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h Donchian channels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 12h volume confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime from chop
        is_choppy = chop_aligned[i] > 61.8  # mean reversion regime
        is_trending = chop_aligned[i] < 38.2  # trend following regime
        
        # Weekly trend filter from daily EMA200
        weekly_uptrend = close[i] > ema200_1d_aligned[i]
        weekly_downtrend = close[i] < ema200_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if is_choppy:
                # In choppy market, exit at Donchian mean (midpoint)
                exit_level = (highest_high[i] + lowest_low[i]) / 2.0
                if close[i] < exit_level:
                    position = 0
                    signals[i] = 0.0
            else:
                # In trending market, exit on Donchian lower band break
                if close[i] < lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if is_choppy:
                # In choppy market, exit at Donchian mean (midpoint)
                exit_level = (highest_high[i] + lowest_low[i]) / 2.0
                if close[i] > exit_level:
                    position = 0
                    signals[i] = 0.0
            else:
                # In trending market, exit on Donchian upper band break
                if close[i] > highest_high[i]:
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if is_choppy:
                    # Mean reversion in choppy market
                    if weekly_uptrend and close[i] <= lowest_low[i]:
                        # Long at support in uptrend chop
                        position = 1
                        signals[i] = 0.25
                    elif weekly_downtrend and close[i] >= highest_high[i]:
                        # Short at resistance in downtrend chop
                        position = -1
                        signals[i] = -0.25
                else:
                    # Trend following in trending market
                    if weekly_uptrend and close[i] > highest_high[i]:
                        # Breakout long in uptrend
                        position = 1
                        signals[i] = 0.25
                    elif weekly_downtrend and close[i] < lowest_low[i]:
                        # Breakdown short in downtrend
                        position = -1
                        signals[i] = -0.25
    
    return signals