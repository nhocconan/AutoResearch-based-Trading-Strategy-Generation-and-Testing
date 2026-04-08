#!/usr/bin/env python3
# daily_price_action_trend_v1
# Hypothesis: Uses daily open/close price action with 1w trend filter and volume confirmation.
# Goes long when daily close > open (bullish candle) in weekly uptrend with above-average volume.
# Goes short when daily close < open (bearish candle) in weekly downtrend with above-average volume.
# Target: 10-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_price_action_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly trend: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: weekly trend changes to downtrend
            if not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend changes to uptrend
            if not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Bullish daily candle: close > open
                bullish_candle = close[i] > open_price[i]
                # Bearish daily candle: close < open
                bearish_candle = close[i] < open_price[i]
                
                # Long entry: bullish candle in weekly uptrend
                if weekly_uptrend and bullish_candle:
                    position = 1
                    signals[i] = 0.25
                # Short entry: bearish candle in weekly downtrend
                elif weekly_downtrend and bearish_candle:
                    position = -1
                    signals[i] = -0.25
    
    return signals