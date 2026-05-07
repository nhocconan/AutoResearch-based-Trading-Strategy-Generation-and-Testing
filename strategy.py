#!/usr/bin/env python3
name = "4h_PriceAction_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily trend filter: EMA34
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Daily range for support/resistance levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_range = daily_high - daily_low
    
    # Dynamic support/resistance: 50% of daily range from close
    support_level = daily_close - 0.5 * daily_range
    resistance_level = daily_close + 0.5 * daily_range
    
    # Align levels to 4h timeframe
    support_aligned = align_htf_to_ltf(prices, df_1d, support_level)
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance_level)
    
    # Volume confirmation: current volume > 1.5x 4-period average
    vol_ma_4 = np.full(n, np.nan)
    for i in range(4, n):
        vol_ma_4[i] = np.mean(volume[i-4:i])
    volume_filter = volume > (1.5 * vol_ma_4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~1.5 days (3*4h) to prevent overtrading
    
    start_idx = max(4, 34)  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(support_aligned[i]) or 
            np.isnan(resistance_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above resistance with volume in daily uptrend
            if (close[i] > resistance_aligned[i] and 
                trending_up and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below support with volume in daily downtrend
            elif (close[i] < support_aligned[i] and 
                  trending_down and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below support or daily trend changes to down
            if close[i] < support_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above resistance or daily trend changes to up
            if close[i] > resistance_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 4h timeframe, price breaking above/below dynamic support/resistance levels (50% of daily range) with volume confirmation and daily EMA34 trend filter captures institutional breakout momentum. The support/resistance levels adapt to daily volatility, making them relevant in both ranging and trending markets. Volume filter ensures institutional participation. Cooldown period prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drift. Works in bull markets (breakouts above resistance in daily uptrend) and bear markets (breakdowns below support in daily downtrend). Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias.