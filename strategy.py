#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w EMA trend filter
# Donchian breakout provides clear structure with proven edge in crypto
# Volume spike confirms institutional participation, reducing false breakouts
# 1w EMA50 > EMA200 filters for bull/bear regime, avoiding counter-trend trades
# Designed for low frequency (75-200 trades over 4 years) with discrete sizing
# Works in both bull and bear: EMA regime filter adapts to market direction, volume confirms legitimacy

name = "4h_Donchian20_1dVolume_1wEMA_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w HTF data for regime filter (EMA crossover)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    # 1w EMA trend filter: EMA50 > EMA200 for bull, EMA50 < EMA200 for bear
    close_1w = pd.Series(df_1w['close'].values)
    ema_50 = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend_bull = ema_50 > ema_200  # Bull regime: EMA50 above EMA200
    ema_trend_bear = ema_50 < ema_200  # Bear regime: EMA50 below EMA200
    ema_trend_aligned_bull = align_htf_to_ltf(prices, df_1w, ema_trend_bull)
    ema_trend_aligned_bear = align_htf_to_ltf(prices, df_1w, ema_trend_bear)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20, 200)  # Need Donchian and EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_trend_aligned_bull[i]) or np.isnan(ema_trend_aligned_bear[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian high with volume spike in bull regime
            if ema_trend_aligned_bull[i] and close[i] > highest_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike in bear regime
            elif ema_trend_aligned_bear[i] and close[i] < lowest_low[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian low or opposite breakout in bear regime
            if close[i] <= lowest_low[i] or (ema_trend_aligned_bear[i] and close[i] < lowest_low[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian high or opposite breakout in bull regime
            if close[i] >= highest_high[i] or (ema_trend_aligned_bull[i] and close[i] > highest_high[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals