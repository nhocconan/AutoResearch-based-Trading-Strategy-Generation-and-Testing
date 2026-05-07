#!/usr/bin/env python3
name = "6h_Donchian20_Breakout_1wTrend_VolumeSurge"
timeframe = "6h"
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
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly EMA40 trend
    ema_40_1w = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    trend_up = close > ema_40_1w_aligned
    trend_down = close < ema_40_1w_aligned
    
    # Donchian channel (20-period) - using weekly high/low for breakout levels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume surge filter: current volume > 2.5x 12-period average (3-day equivalent in 6h)
    vol_ma_12 = np.full(n, np.nan)
    for i in range(12, n):
        vol_ma_12[i] = np.mean(volume[i-12:i])
    vol_surge = volume > (2.5 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~1 day (3*6h) to prevent overtrading
    
    start_idx = max(12, 20, 40)  # Ensure enough data for volume MA, Donchian, and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_40_1w_aligned[i])):
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
            # Long: Price breaks above Donchian high with volume surge in weekly uptrend
            if (close[i] > donchian_high_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Donchian low with volume surge in weekly downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Donchian low or weekly trend changes to down
            if close[i] < donchian_low_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Donchian high or weekly trend changes to up
            if close[i] > donchian_high_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, price breaking above/below 20-period Donchian channel with volume surge confirmation and weekly EMA40 trend filter captures institutional breakout momentum. The Donchian channel represents key support/resistance levels, while the weekly trend filter ensures alignment with higher timeframe momentum. Volume surge filter (2.5x 12-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull markets (breakouts above Donchian high in weekly uptrend) and bear markets (breakdowns below Donchian low in weekly downtrend). Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This approach combines price action breakouts with trend and volume confirmation, addressing the limitations of pure Donchian strategies that suffered from false breakouts in ranging markets.