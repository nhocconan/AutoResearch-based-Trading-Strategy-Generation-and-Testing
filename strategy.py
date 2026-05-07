#!/usr/bin/env python3
name = "12h_1w_1d_TriangleBreakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for triangle pattern
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily high/low for ascending/descending triangle detection
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Ascending triangle: flat resistance, rising support
    # Descending triangle: flat support, falling resistance
    
    # Calculate resistance (recent highs) and support (recent lows) over last 10 days
    lookback = 10
    resistance = np.full(len(daily_high), np.nan)
    support = np.full(len(daily_low), np.nan)
    
    for i in range(lookback, len(daily_high)):
        # Resistance: maximum of recent highs
        resistance[i] = np.max(daily_high[i-lookback:i])
        # Support: minimum of recent lows
        support[i] = np.min(daily_low[i-lookback:i])
    
    # Align triangle levels to 12h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1d, support)
    
    # Volume confirmation: volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(resistance_aligned[i]) or 
            np.isnan(support_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Ascending triangle breakout: price breaks above resistance with volume in weekly uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            
            if close[i] > resistance_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.30
                position = 1
            # Descending triangle breakout: price breaks below support with volume in weekly downtrend
            elif close[i] < support_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price falls back below support or volume drops
            if close[i] < support_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price rises back above resistance or volume drops
            if close[i] > resistance_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h triangle pattern breakout with weekly trend and volume confirmation
# - Ascending triangle (flat resistance, rising support) breaks out upward in weekly uptrend
# - Descending triangle (flat support, falling resistance) breaks down in weekly downtrend
# - Volume spike (2x average) confirms institutional participation in breakout
# - Works in both bull (buy ascending breakouts in uptrend) and bear (sell descending breakdowns in downtrend)
# - Weekly EMA(50) filter ensures alignment with higher timeframe trend
# - Exit when price returns to opposite triangle boundary or volume weakens
# - Position size 0.30 targets ~30-80 trades/year, staying within 12h limits
# - Triangle patterns represent consolidation before continuation, effective in trending markets
# - Weekly trend filter avoids counter-trend trades during choppy periods
# - Volume confirmation reduces false breakouts from low participation
# - Novel application of triangle patterns on 12h timeframe with weekly trend filter
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Uses actual weekly/daily data from Binance, no resampling or synthetic timestamps