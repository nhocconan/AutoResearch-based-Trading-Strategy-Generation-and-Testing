#!/usr/bin/env python3
# 4h_donchian_breakout_1d_volume_chop_v4
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price vs 1d EMA50) and volume confirmation + choppiness regime filter.
# Works in bull/bear: Donchian breakouts capture momentum; 1d EMA50 ensures alignment with higher timeframe trend; volume confirms breakout validity; chop filter avoids whipsaws in ranging markets.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_chop_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channels (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # 4h choppiness index regime filter (14-period)
    chop_period = 14
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=1, min_periods=1).sum()  # True Range
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    
    # 4h volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish OR chop > 61.8 (strong ranging)
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish OR chop > 61.8 (strong ranging)
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and chop < 61.8 (not strongly ranging)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            chop_filter = chop[i] < 61.8
            
            if volume_confirmed and chop_filter:
                # Long: price breaks above Donchian high + above 1d EMA50 (uptrend)
                if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low + below 1d EMA50 (downtrend)
                elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals