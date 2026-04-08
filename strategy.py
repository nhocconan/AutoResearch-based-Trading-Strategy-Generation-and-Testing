#!/usr/bin/env python3
# 1d_weekly_ema_breakout_volume
# Hypothesis: Uses 1-day Exponential Moving Average (EMA200) for trend filter with weekly EMA20 for momentum confirmation and volume spike on breakout.
# Enters long when price breaks above weekly EMA20 in a daily uptrend with volume confirmation; short when price breaks below weekly EMA20 in a daily downtrend with volume confirmation.
# Exits on opposite weekly EMA crossover or trend reversal. Designed for low trade frequency (~10-25/year) to minimize fee drift.
# Uses weekly EMA for stronger trend filter to reduce whipsaw and improve performance in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_ema_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend and momentum filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for momentum
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily EMA200 for trend filter
    ema200_daily = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation (20-day average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Wait for daily EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(ema200_daily[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema200_daily[i]
        daily_downtrend = close[i] < ema200_daily[i]
        
        # Weekly EMA crossover signals
        weekly_bullish = close[i] > ema20_1w_aligned[i]
        weekly_bearish = close[i] < ema20_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below weekly EMA20 or trend change
            if weekly_bearish or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above weekly EMA20 or trend change
            if weekly_bullish or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: Price above weekly EMA20 in daily uptrend
                if daily_uptrend and weekly_bullish:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Price below weekly EMA20 in daily downtrend
                elif daily_downtrend and weekly_bearish:
                    position = -1
                    signals[i] = -0.25
    
    return signals