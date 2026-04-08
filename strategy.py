# 1d_2025_scalp_v1
# Hypothesis: In 2025 bearish/range markets, price often reverses sharply after reaching weekly extremes. Strategy identifies weekly Donchian channel extremes on 1d timeframe and enters on mean-reversion bounces with volume confirmation. Uses weekly trend filter to avoid counter-trend trades. Target: 10-25 trades/year to minimize fee drag in choppy conditions.

name = "1d_2025_scalp_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate Donchian channels
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Weekly trend filter: EMA(34)
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    vol_ma[19:] = pd.Series(volume).rolling(window=20, min_periods=20).mean()[19:].values
    
    # Start from sufficient lookback
    start_idx = 40  # Need 20 for Donchian + buffer
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches weekly EMA (mean reversion target) or stops make new low
            if close[i] >= weekly_ema_aligned[i] or low[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches weekly EMA or stops make new high
            if close[i] <= weekly_ema_aligned[i] or high[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Mean reversion long: price touches weekly Donchian low in downtrend
                if low[i] <= donchian_low_aligned[i] and close[i] < weekly_ema_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short: price touches weekly Donchian high in uptrend
                elif high[i] >= donchian_high_aligned[i] and close[i] > weekly_ema_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals