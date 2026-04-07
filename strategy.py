#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) breakout with weekly volume confirmation and weekly trend filter
# Hypothesis: Donchian breakouts capture momentum moves; volume confirms institutional participation.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Works in bull via upward breakouts, in bear via downward breakdowns. Target: 15-35 trades/year.
name = "12h_donchian20_weekly_volume_trend_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume and trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly 20-period volume moving average
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate weekly EMA(50) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian(20) channels for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > weekly average volume
        vol_confirm = volume[i] > vol_ma_1w_aligned[i]
        
        # Trend filter: price above/below weekly EMA(50)
        price_above_weekly_ema = close[i] > ema_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (trend reversal)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (trend reversal)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high + volume confirmation + uptrend filter
            if (close[i] > donchian_high[i] and vol_confirm and price_above_weekly_ema):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low + volume confirmation + downtrend filter
            elif (close[i] < donchian_low[i] and vol_confirm and price_below_weekly_ema):
                position = -1
                signals[i] = -0.25
    
    return signals