#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1wTrend_VolumeRegime
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high AND weekly trend is up (close > 1w EMA50) AND volume > 1.5x 20-period average AND choppiness index < 45 (trending regime). Enter short when price breaks below 20-period Donchian low AND weekly trend is down (close < 1w EMA50) AND volume spike AND chop < 45. Exit on Donchian opposite breakout or weekly trend reversal. Uses weekly EMA50 for smooth trend filter and chop filter to avoid whipsaw in ranging markets. Targets 20-50 trades/year on BTC/ETH/SOL with controlled fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    chop_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(chop_sum / np.maximum(highest_high - lowest_low, 1e-10)) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when not enough data
    chop_filter = chop < 45  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian, volume MA, ATR, and chop warmup
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Weekly trend filter
        trend_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + weekly uptrend + trending chop
            long_signal = breakout_up and volume_spike[i] and trend_uptrend and chop_filter[i]
            
            # Short: Donchian breakout down + volume spike + weekly downtrend + trending chop
            short_signal = breakout_down and volume_spike[i] and trend_downtrend and chop_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Donchian breakout down OR weekly trend change to downtrend
            if breakout_down or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Donchian breakout up OR weekly trend change to uptrend
            if breakout_up or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1wTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0