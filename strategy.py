#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian channels identify key support/resistance levels. Break of upper/lower band with
# 12h EMA50 trend alignment and volume spike captures strong momentum moves.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.
# Works in bull markets (breakouts continue trend) and bear markets (breakdowns continue downtrend).

name = "4h_Donchian20_12hEMA50_Volume_v1"
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
    
    # Align Donchian levels to 4h timeframe (wait for 12h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = donchian_high_aligned[i]
        curr_low = donchian_low_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 12h trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above upper band + price above 12h EMA50
                if curr_close > curr_high and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Close breaks below lower band + price below 12h EMA50
                elif curr_close < curr_low and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian middle or loses 12h trend
            donchian_mid = (curr_high + curr_low) / 2
            if curr_close < donchian_mid or curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian middle or loses 12h trend
            donchian_mid = (curr_high + curr_low) / 2
            if curr_close > donchian_mid or curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals