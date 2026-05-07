#!/usr/bin/env python3
name = "1d_1w_Donchian_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Weekly trend filter: EMA(50) on weekly close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume spike detection: 5-period average
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 5)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_5[i] * 2.0
            uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            
            if close[i] > high_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume and weekly downtrend
            elif close[i] < low_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below weekly low or volume drops
            if close[i] < low_20_aligned[i] or volume[i] < vol_ma_5[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above weekly high or volume drops
            if close[i] > high_20_aligned[i] or volume[i] < vol_ma_5[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Donchian breakout with weekly trend and volume confirmation
# - Weekly Donchian (20-period) acts as major support/resistance
# - Breakout above weekly high with volume in weekly uptrend = long opportunity
# - Breakdown below weekly low with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Weekly EMA(50) filter ensures we trade with the higher timeframe trend
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to weekly low (for longs) or high (for shorts) or volume weakens
# - Position size 0.25 targets ~15-25 trades/year, avoiding fee drag on daily timeframe
# - Uses actual weekly Donchian channels from Binance data (no resampling)