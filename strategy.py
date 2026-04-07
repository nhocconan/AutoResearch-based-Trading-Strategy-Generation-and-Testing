#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout with 1w Trend Filter and Volume Confirmation
# Hypothesis: Donchian channel breakouts on daily timeframe capture strong momentum moves.
# Combined with 1-week EMA20 trend filter to avoid counter-trend trades and align with major trends.
# Volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
# Targets 10-20 trades/year with disciplined entries to avoid overtrading.

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 20-day Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day average volume for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for Donchian and volume SMA
        # Skip if required data not available
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1w EMA20 OR price breaks below 10-day Donchian mid-point
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < ema20_1w_aligned[i] or close[i] < donchian_mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price closes above 1w EMA20 OR price breaks above 10-day Donchian mid-point
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > ema20_1w_aligned[i] or close[i] > donchian_mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above 20-day Donchian high + volume confirmation + uptrend
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short: price breaks below 20-day Donchian low + volume confirmation + downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.30
    
    return signals