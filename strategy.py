# Based on the provided experiment context, here is a new strategy that avoids overtrading and focuses on proven patterns:
# - Uses 4h timeframe as required
# - Implements Donchian breakout with volume confirmation and trend filter (proven pattern)
# - Adds volatility filter to reduce false signals
# - Designed for low trade frequency (target: 20-50 trades/year)
# - Works in both bull and bear markets via trend filter

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Long when price breaks above 20-period high AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below 20-period low AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses 10-period EMA (opposite direction)
# Donchian provides clear breakout levels, EMA50 filters trend direction, volume confirms conviction
# Target: 25-40 trades/year by requiring all three conditions simultaneously

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period high and low for Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donchian_high = high_20[i]
        donchian_low = low_20[i]
        ema50_val = ema50_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Get current 1d volume (same for all 4h bars within the day)
        idx_1d = i // 6  # 6 bars per day (24h/4h)
        if idx_1d >= len(df_1d):
            idx_1d = len(df_1d) - 1
        volume = df_1d['volume'].iloc[idx_1d] if idx_1d >= 0 else df_1d['volume'].iloc[0]
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > EMA50 AND volume confirmation
            if price > donchian_high and price > ema50_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < EMA50 AND volume confirmation
            elif price < donchian_low and price < ema50_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below 10-period EMA
                if price < ema10[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above 10-period EMA
                if price > ema10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0