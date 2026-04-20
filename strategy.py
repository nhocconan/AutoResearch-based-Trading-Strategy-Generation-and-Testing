#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with weekly trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high + price above weekly EMA50 + volume > 1.5x average
# - Short when price breaks below Donchian(20) low + price below weekly EMA50 + volume > 1.5x average
# - Exit when price returns to Donchian midpoint or trend reverses
# - Weekly EMA50 provides strong trend filter to avoid counter-trend trades
# - Volume confirmation reduces false breakouts
# - Designed for 4h timeframe with selective entries to target 20-50 trades per year per symbol

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate Donchian channels on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    donchian_period = 20
    donchian_high = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate average volume for confirmation
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume_4h[i] > 1.5 * vol_ma[i]
        
        # Trend filter
        price_above_weekly_ema = close_4h[i] > ema50_weekly_aligned[i]
        price_below_weekly_ema = close_4h[i] < ema50_weekly_aligned[i]
        
        if position == 0:
            # Long entry: break above Donchian high + above weekly EMA + volume confirmation
            if close_4h[i] > donchian_high[i] and price_above_weekly_ema and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low + below weekly EMA + volume confirmation
            elif close_4h[i] < donchian_low[i] and price_below_weekly_ema and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or trend reverses
            if close_4h[i] <= donchian_mid[i] or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or trend reverses
            if close_4h[i] >= donchian_mid[i] or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_WeeklyEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0