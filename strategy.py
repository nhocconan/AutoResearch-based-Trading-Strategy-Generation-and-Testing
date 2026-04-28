#!/usr/bin/env python3
"""
4h_RSI_Divergence_Volume_Filter
Hypothesis: RSI divergence (hidden divergence) on 4h with volume confirmation and 1d trend filter.
Hidden bullish: price makes higher low, RSI makes lower low → long in uptrend.
Hidden bearish: price makes lower high, RSI makes higher high → short in downtrend.
Targets 20-30 trades/year to minimize fee drag while capturing trend continuations.
Works in both bull and bear markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(rsi_values[i-1]) or np.isnan(vol_ma_20[i]) or
            i < 2):  # Need at least 2 bars for divergence check
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Check for hidden bullish divergence: price higher low, RSI lower low
        # Need to look back for swing lows
        hidden_bull = False
        hidden_bear = False
        
        # Simple swing detection: look for local lows/highs over 3 bars
        if i >= 3:
            # Price low at i-1
            if low[i-1] <= low[i-2] and low[i-1] <= low[i]:
                # RSI lower low at i-1
                if rsi_values[i-1] <= rsi_values[i-2] and rsi_values[i-1] <= rsi_values[i]:
                    hidden_bull = True
            
            # Price high at i-1
            if high[i-1] >= high[i-2] and high[i-1] >= high[i]:
                # RSI higher high at i-1
                if rsi_values[i-1] >= rsi_values[i-2] and rsi_values[i-1] >= rsi_values[i]:
                    hidden_bear = True
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: hidden divergence in direction of 1d trend
        long_entry = vol_confirm and trend_up and hidden_bull
        short_entry = vol_confirm and trend_down and hidden_bear
        
        # Exit logic: opposite divergence or trend change
        long_exit = hidden_bear or not trend_up
        short_exit = hidden_bull or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_Divergence_Volume_Filter"
timeframe = "4h"
leverage = 1.0