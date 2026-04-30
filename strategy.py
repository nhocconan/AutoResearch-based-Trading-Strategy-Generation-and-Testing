#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian(20) provides robust price channel structure that works in both bull and bear markets
# 1w EMA50 ensures we only trade with the major weekly trend, reducing whipsaws
# Volume spike (2.0x 20-period average) confirms institutional participation and reduces false breakouts
# Exit on opposite Donchian(10) touch for faster profit-taking and reduced drawdown
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 4h Donchian channels (20 for entry, 10 for exit)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian(20) for entry
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit
    donchian_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    
    # Align 4h Donchian levels to lower timeframe (if needed, but here primary is 4h)
    # Since we're on 4h timeframe, no alignment needed for 4h indicators
    donchian_high_20_aligned = donchian_high_20
    donchian_low_20_aligned = donchian_low_20
    donchian_high_10_aligned = donchian_high_10
    donchian_low_10_aligned = donchian_low_10
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50, 20)  # warmup for EMA50, Donchian20, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(donchian_high_10_aligned[i]) or 
            np.isnan(donchian_low_10_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_donchian_high_20 = donchian_high_20_aligned[i]
        curr_donchian_low_20 = donchian_low_20_aligned[i]
        curr_donchian_high_10 = donchian_high_10_aligned[i]
        curr_donchian_low_10 = donchian_low_10_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above Donchian(20) high AND above 1w EMA50 (uptrend)
                if curr_high > curr_donchian_high_20 and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Donchian(20) low AND below 1w EMA50 (downtrend)
                elif curr_low < curr_donchian_low_20 and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price touches Donchian(10) low (faster exit for profit protection)
            if curr_low <= curr_donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price touches Donchian(10) high (faster exit for profit protection)
            if curr_high >= curr_donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals