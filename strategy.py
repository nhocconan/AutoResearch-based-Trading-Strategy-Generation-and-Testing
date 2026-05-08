#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate 20-period high and low for weekly
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Weekly trend: EMA34 on weekly close
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_daily = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Daily volume filter: volume > 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for position sizing (optional - using fixed size per rules)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or
            np.isnan(ema34_weekly_daily[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + above weekly EMA34 + volume surge
            long_cond = (close[i] > donchian_high_daily[i] and 
                        close[i] > ema34_weekly_daily[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: price breaks below weekly Donchian low + below weekly EMA34 + volume surge
            short_cond = (close[i] < donchian_low_daily[i] and 
                         close[i] < ema34_weekly_daily[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low
            if close[i] < donchian_low_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high
            if close[i] > donchian_high_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with trend filter and volume confirmation.
# Enters long when daily price breaks above weekly Donchian high (20-period) 
# while above weekly EMA34 (trend filter) with volume confirmation.
# Enters short when price breaks below weekly Donchian low while below weekly EMA34.
# Exits on opposite Donchian break. Designed to capture major trends while
# avoiding false breakouts in low volume/ranging markets. Weekly timeframe
# reduces noise and false signals. Target: 15-30 trades/year to minimize fee drag.
# Works in both bull (trend following) and bear (trend following) markets by
# capturing significant breakdowns as well as breakouts. Uses discrete sizing (0.25) 
# to reduce churn. Weekly Donchian provides structural support/resistance levels
# that institutions watch. Volume surge confirms institutional participation.