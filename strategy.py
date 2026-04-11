#!/usr/bin/env python3
# 12h_1d_donchian_breakout_v1
# Strategy: 12h Donchian(20) breakout with 1d volume confirmation and RSI filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture breakout momentum. Volume confirmation ensures institutional
# participation. RSI filter avoids overbought/oversold extremes. Designed for low trade frequency (~15-30/year)
# to minimize fee drag. Works in bull markets via upside breakouts and bear markets via downside breakdowns.
# Uses 1d volume average and RSI for confirmation to reduce false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h RSI(14) for filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(rsi[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # RSI filter: avoid extremes (30 < RSI < 70)
        rsi_filter = (rsi[i] > 30) & (rsi[i] < 70)
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i-1]  # break above prior 20-period high
        breakdown_down = close[i] < low_20[i-1]  # break below prior 20-period low
        
        # Entry conditions
        # Long: upward breakout AND volume confirmation AND RSI filter
        if breakout_up and vol_confirm and rsi_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: downward breakdown AND volume confirmation AND RSI filter
        elif breakdown_down and vol_confirm and rsi_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout (trend reversal)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals