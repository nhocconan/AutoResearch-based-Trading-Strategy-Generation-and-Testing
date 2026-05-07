#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation.
# Uses weekly EMA200 for trend direction, 1d Donchian channels for entry, and volume spikes for confirmation.
# Designed to work in both bull and bear markets by following the weekly trend direction.
# Target: 10-30 trades/year per symbol to avoid excessive fee drift.
name = "1d_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Weekly trend filter: 200-period EMA on close
    ema_200_weekly = pd.Series(df_weekly['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_200_weekly)
    
    # Daily Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.where(vol_ma_20 > 0, volume / vol_ma_20, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_weekly_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA200
        uptrend = close[i] > ema_200_weekly_aligned[i]
        downtrend = close[i] < ema_200_weekly_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume spike in uptrend
            long_condition = (close[i] > donchian_upper[i]) and vol_spike[i] and uptrend
            # Short breakdown: price breaks below lower Donchian with volume spike in downtrend
            short_condition = (close[i] < donchian_lower[i]) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below upper Donchian or trend turns down
            if (close[i] < donchian_upper[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above lower Donchian or trend turns up
            if (close[i] > donchian_lower[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals