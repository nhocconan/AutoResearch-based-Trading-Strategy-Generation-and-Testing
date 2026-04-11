#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Bollinger Band squeeze + daily trend confirmation.
# Uses weekly Bollinger Band width percentile to detect low volatility (squeeze) conditions.
# When squeeze occurs (<20th percentile), trades in direction of daily EMA(50) trend.
# Volume filter confirms breakout validity.
# Designed for 12-30 trades/year to minimize fee drift while capturing explosive moves after consolidation.
# Works in bull/bear markets by trading breakouts from squeezes regardless of trend direction.

name = "6h_1w_bb_squeeze_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Weekly BB width percentile (lookback 50 weeks)
    bb_width_pct = np.full_like(bb_width, np.nan, dtype=float)
    for i in range(49, len(bb_width)):
        window = bb_width[i-49:i+1]
        if not np.any(np.isnan(window)):
            bb_width_pct[i] = (np.sum(window <= bb_width[i]) / len(window)) * 100
    
    # Daily EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align weekly indicators to 6h timeframe
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1w, bb_width_pct)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_width_pct_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Squeeze condition: BB width < 20th percentile (low volatility)
        squeeze = bb_width_pct_aligned[i] < 20
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Trend condition: price relative to daily EMA(50)
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Entry logic: breakout from squeeze in trend direction
        long_entry = squeeze and above_ema and vol_filter
        short_entry = squeeze and below_ema and vol_filter
        
        # Exit conditions: opposite squeeze or loss of volatility
        long_exit = not squeeze or (close[i] < ema_50_aligned[i])
        short_exit = not squeeze or (close[i] > ema_50_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals