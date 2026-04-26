#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_FundingMeanRev_v1
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and funding rate mean reversion (z-score < -2 long, > +2 short). 
Works in bull/bear via 1d trend alignment and funding extremes. Targets 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25).
Uses volume confirmation (1.5x avg) to reduce false breakouts. Funding data loaded from processed/funding/ directory.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop for trend filter and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Donchian channels from 1d OHLC (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Load funding rate data (8h intervals) and compute z-score
    try:
        funding_df = pd.read_parquet('/mnt/data/funding/BTCUSDT-funding-8h.parquet')
        funding_df = funding_df.set_index('open_time')
        # Align funding to 6h prices using forward fill (funding updates every 8h)
        funding_series = pd.Series(index=prices.index, dtype=float)
        funding_series.update(funding_df['funding_rate'])
        funding_series = funding_series.ffill()
        funding_values = funding_series.values
        
        # Calculate 30-day z-score of funding rate (90 samples for 8h data)
        funding_ma = pd.Series(funding_values).rolling(window=90, min_periods=90).mean().values
        funding_std = pd.Series(funding_values).rolling(window=90, min_periods=90).std().values
        funding_z = np.where(funding_std > 0, (funding_values - funding_ma) / funding_std, 0)
        
        # Funding signals: long when z < -2, short when z > +2
        funding_long = funding_z < -2.0
        funding_short = funding_z > 2.0
    except:
        # Fallback if funding data not available
        funding_long = np.zeros(n, dtype=bool)
        funding_short = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian/volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume, trend, and funding confirmation
        if position == 0:
            # Long: Price breaks above Donchian high AND 1d uptrend AND volume spike AND funding extreme long
            if close[i] > donchian_high_aligned[i] and trend_1d[i] == 1 and volume_spike[i] and funding_long[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1d downtrend AND volume spike AND funding extreme short
            elif close[i] < donchian_low_aligned[i] and trend_1d[i] == -1 and volume_spike[i] and funding_short[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 1d trend turns down OR funding normalizes
            if close[i] < donchian_low_aligned[i] or trend_1d[i] == -1 or (funding_z[i] > -0.5 and funding_z[i] < 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 1d trend turns up OR funding normalizes
            if close[i] > donchian_high_aligned[i] or trend_1d[i] == 1 or (funding_z[i] > -0.5 and funding_z[i] < 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_FundingMeanRev_v1"
timeframe = "6h"
leverage = 1.0