#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction + 1d EMA50 trend filter + volume confirmation
# Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) to reduce noise.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag while capturing trends.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

name = "1h_Donchian20_4hTrend_1dEMA50_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Session filter: 08-20 UTC (precompute hour array)
    hours = prices.index.hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian(20) for direction - using completed 4h bars only
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_4h_upper = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    donchian_4h_lower = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)  # warmup for EMA50, Donchian, volume
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(donchian_4h_upper[i]) or np.isnan(donchian_4h_lower[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_upper = donchian_4h_upper[i]
        curr_lower = donchian_4h_lower[i]
        
        # Exit conditions: trend reversal or Donchian breakdown in opposite direction
        if position == 1:  # Long position
            if (curr_close < curr_ema_50_1d or  # trend change
                curr_close < curr_lower):       # Donchian breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            if (curr_close > curr_ema_50_1d or  # trend change
                curr_close > curr_upper):       # Donchian breakout
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries with session and volume confirmation
            # Long entry: price above 4h Donchian upper + above 1d EMA50 + volume
            if (curr_close > curr_upper and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short entry: price below 4h Donchian lower + below 1d EMA50 + volume
            elif (curr_close < curr_lower and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals