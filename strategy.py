#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel provides robust price structure for breakouts
# 1w EMA > 50 ensures we trade only in primary trend direction, avoiding counter-trend whipsaws
# Volume spike confirms institutional participation behind the breakout
# Designed for very low frequency (30-100 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + price structure logic

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian(20) channels from prior 20 periods
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift to avoid look-ahead: use prior 20-period high/low
    prior_high_20 = np.concatenate([[np.nan], high_20[:-1]])
    prior_low_20 = np.concatenate([[np.nan], low_20[:-1]])
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need 1w EMA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(prior_high_20[i]) or np.isnan(prior_low_20[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > prior_high_20[i]  # Price breaks above prior 20-period high
        breakout_short = close[i] < prior_low_20[i]  # Price breaks below prior 20-period low
        
        # Trend filter: price above/below 1w EMA50
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above prior high with volume spike and above weekly EMA
            if breakout_long and vol_spike and above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below prior low with volume spike and below weekly EMA
            elif breakout_short and vol_spike and below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior 20-period low or price crosses below weekly EMA
            if close[i] < prior_low_20[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above prior 20-period high or price crosses above weekly EMA
            if close[i] > prior_high_20[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals