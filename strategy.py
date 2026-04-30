#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels identify key support/resistance levels. Break of upper/lower band with
# 1d EMA34 trend alignment and volume spike captures strong momentum moves.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# Works in bull markets (breakouts continue trend) and bear markets (breakdowns continue downtrend).

name = "12h_Donchian20_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels from prior 12h bar
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Prior 12h OHLC for Donchian calculation (shift to avoid look-ahead)
    prior_high = df_12h['high'].shift(1).values
    prior_low = df_12h['low'].shift(1).values
    
    # Calculate Donchian(20) on 12h timeframe
    high_20 = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for 12h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = donchian_high_aligned[i]
        curr_low = donchian_low_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1d trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above upper band + price above 1d EMA34
                if curr_close > curr_high and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Close breaks below lower band + price below 1d EMA34
                elif curr_close < curr_low and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian middle or loses 1d trend
            donchian_mid = (curr_high + curr_low) / 2
            if curr_close < donchian_mid or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian middle or loses 1d trend
            donchian_mid = (curr_high + curr_low) / 2
            if curr_close > donchian_mid or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals