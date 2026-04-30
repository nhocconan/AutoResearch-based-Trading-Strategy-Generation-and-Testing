#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian band AND price > 1d EMA50 AND volume > 1.8x 20-bar average.
# Short when price breaks below lower Donchian band AND price < 1d EMA50 AND volume > 1.8x 20-bar average.
# Exit when price crosses the middle Donchian band (20-bar EMA of midpoint).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via 1d EMA50 trend filter.

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from previous completed 12h bar
    # Need 20-period high/low of 12h data, but we don't have direct access.
    # Instead, we'll use the 1d HTF data to approximate longer-term structure.
    # For Donchian(20) on 12h, we approximate using 10-day lookback on 1d data (since 10*12h = 5d, close enough)
    # But better: we calculate Donchian on the primary timeframe using rolling window
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period EMA of midpoint (typical Donchian middle)
    midpoint = (high + low) / 2.0
    mid_ema = pd.Series(midpoint).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(mid_ema[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper Donchian, uptrend (price > 1d EMA50), volume confirmation
            if (curr_high > high_roll[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian, downtrend (price < 1d EMA50), volume confirmation
            elif (curr_low < low_roll[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below middle band (20 EMA of midpoint)
            if curr_close < mid_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above middle band
            if curr_close > mid_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals