#!/usr/bin/env python3
name = "1d_1w_1wTrend_Volume_DonchianBreakout"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend and Donchian
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Weekly Donchian(20) channels for breakout signals
    highest_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: 3-day average (3x 1d bars)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 3)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > donchian_high_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian low with volume and weekly downtrend
            elif close[i] < donchian_low_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns below weekly Donchian high or volume drops
            if close[i] < donchian_high_aligned[i] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns above weekly Donchian low or volume drops
            if close[i] > donchian_low_aligned[i] or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 1d Weekly Donchian(20) breakout with weekly trend and volume confirmation
# - Weekly Donchian channels provide major support/resistance levels
# - Breakout above weekly high with volume in weekly uptrend = long opportunity
# - Breakdown below weekly low with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x 3-day average) confirms institutional participation
# - Weekly EMA(34) trend filter reduces whipsaws and works in both bull/bear markets
# - Exit when price returns to weekly Donchian level or volume weakens
# - Position size 0.30 targets ~15-25 trades/year, well within limits
# - Weekly timeframe avoids daily noise while capturing major trends
# - Works in BOTH bull (buy weekly high breaks in uptrend) and bear (sell weekly low breaks in downtrend) markets
# - Volume confirmation reduces false breakouts from low-volume spikes
# - Novel: Weekly Donchian breakout with volume + trend filter (not recently tried on 1d)
# - Aims for 60-100 total trades over 4 years (15-25/year) to minimize fee drag