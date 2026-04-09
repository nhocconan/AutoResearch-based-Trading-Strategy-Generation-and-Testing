#!/usr/bin/env python3
# 12h_donchian_breakout_1w_ema_volume_v1
# Hypothesis: 12h Donchian channel breakout with 1-week EMA trend filter and volume confirmation.
# Works in bull/bear: 1w EMA filters primary trend direction, Donchian breakout captures momentum,
# volume ensures validity. Exits on opposite Donchian breakout or trend reversal.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA(21)
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h Donchian channel (20-period)
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1w_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 12h Donchian low OR trend turns bearish
            if low[i] < low_roll[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12h Donchian high OR trend turns bullish
            if high[i] > high_roll[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Check for Donchian breakouts
                if high[i] > high_roll[i] and close[i] > ema_1w_aligned[i]:
                    # Bullish breakout + price above 1w EMA → long
                    position = 1
                    signals[i] = 0.25
                elif low[i] < low_roll[i] and close[i] < ema_1w_aligned[i]:
                    # Bearish breakout + price below 1w EMA → short
                    position = -1
                    signals[i] = -0.25
    
    return signals