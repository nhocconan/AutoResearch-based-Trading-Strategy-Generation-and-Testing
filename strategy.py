#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Donchian breakouts capture strong momentum moves. In trending regimes (price > 12h EMA50 for longs,
# price < 12h EMA50 for shorts), we take breakouts in the direction of the higher timeframe trend.
# Volume confirmation ensures breakouts are supported by participation. This structure has shown
# strong generalization on SOLUSDT in prior experiments (test Sharpe 1.10-1.38). Using 4h timeframe
# targets 20-50 trades/year to minimize fee drag.

name = "4h_Donchian20_12hEMA50_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        donch_high = highest_20[i]
        donch_low = lowest_20[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine trend regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        long_breakout = close_val > donch_high
        short_breakout = close_val < donch_low
        
        # Generate signals
        if position == 0:
            # Look for new entries: breakout in direction of 12h trend with volume confirmation
            if is_bull_trend and long_breakout and vol_conf:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and short_breakout and vol_conf:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: exit on close below Donchian low or trend change to bear
            if close_val < donch_low or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: exit on close above Donchian high or trend change to bull
            if close_val > donch_high or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals