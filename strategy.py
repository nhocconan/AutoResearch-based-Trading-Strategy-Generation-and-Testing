#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# - Long: price breaks above 20-day Donchian high, weekly EMA21 rising, volume > 1.5x 20-day avg
# - Short: price breaks below 20-day Donchian low, weekly EMA21 falling, volume > 1.5x 20-day avg
# - Exit: price returns to opposite Donchian level (20-day low for long exit, high for short exit)
# - Uses weekly EMA21 for trend filter to avoid counter-trend trades
# - Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# - Works in both bull and bear markets by following weekly trend

name = "1d_donchian_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for EMA21 trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return signals
    
    # Pre-compute weekly EMA21 for trend filter
    weekly_close = df_1w['close'].values
    ema_21 = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Pre-compute 20-day Donchian channels (using prior 20 days for current bar)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Pre-compute 20-day volume average for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Weekly trend filter: EMA21 rising/falling
        ema_rising = ema_21_aligned[i] > ema_21_aligned[i-1] if i > 0 else False
        ema_falling = ema_21_aligned[i] < ema_21_aligned[i-1] if i > 0 else False
        
        # Donchian levels
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price > Donchian high, weekly EMA rising, volume confirmation
        if close_price > donchian_high and ema_rising and vol_confirm:
            enter_long = True
        
        # Short breakout: price < Donchian low, weekly EMA falling, volume confirmation
        if close_price < donchian_low and ema_falling and vol_confirm:
            enter_short = True
        
        # Exit conditions: return to opposite Donchian level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to or below Donchian low
            exit_long = close_price <= donchian_low
        elif position == -1:
            # Exit short if price returns to or above Donchian high
            exit_short = close_price >= donchian_high
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals