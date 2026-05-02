#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly trend filter (price > weekly EMA50) + daily volume confirmation
# Donchian channels provide clear trend-following structure with proven efficacy in crypto
# Weekly EMA50 ensures alignment with major trend to avoid counter-trend trades in both bull/bear markets
# Daily volume spike (>2.0 x 20-period EMA) filters false breakouts
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing (0.25) to control fee churn and drawdown

name = "6h_Donchian20_WeeklyEMA50_Trend_DailyVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume EMA20 for spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Donchian(20) channels on 6h timeframe
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, lookback - 1)  # weekly EMA50 needs 50 bars, Donchian needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.0 x daily average volume
        # Approximate: use aligned daily volume EMA as threshold for 6h bar
        volume_confirmation = volume[i] > (2.0 * vol_ema_20_1d_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper band with volume confirmation and uptrend
            if high[i] > highest_high[i] and volume_confirmation and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower band with volume confirmation and downtrend
            elif low[i] < lowest_low[i] and volume_confirmation and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band (reversal) OR trend changes to downtrend
            if low[i] < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band (reversal) OR trend changes to uptrend
            if high[i] > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals