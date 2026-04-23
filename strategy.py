#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Primary timeframe: 1d for lower trade frequency and better signal quality
- HTF: 1w EMA50 to capture major trend direction (avoids whipsaws in ranging markets)
- Entry: Price breaks above/below 20-day Donchian channel + price >/< 1w EMA50 + volume > 1.5x 20-day average
- Exit: Price returns to middle of Donchian channel (10-day EMA of high/low) OR trend reversal
- Position sizing: 0.25 (discrete level to minimize fee churn)
- Designed for BTC/ETH: works in bull markets via breakouts, in bear markets via trend-filtered shorts
- Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
"""

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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 1d data
    # Highest high over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low over 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle channel (10-period EMA of high/low for smoother exit)
    avg_hl = (high + low) / 2
    donchian_middle = pd.Series(avg_hl).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_1w and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_middle[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above Donchian upper band + uptrend (price > 1w EMA50) + volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian lower band + downtrend (price < 1w EMA50) + volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel OR trend reversal
            exit_signal = False
            if position == 1:
                # Exit long when price < Donchian middle OR price < 1w EMA50 (trend change)
                if close[i] < donchian_middle[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian middle OR price > 1w EMA50 (trend change)
                if close[i] > donchian_middle[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0