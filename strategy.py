#!/usr/bin/env python3
"""
12h_1d_ThreeBar_Pullback_TrendContinuation
Hypothesis: Uses 1d EMA34 for primary trend direction and enters on 12h three-bar pullbacks in trending direction.
Adds volume confirmation to ensure institutional participation. Designed for low trade frequency by requiring
strong trend alignment and pullback structure. Works in bull markets by buying dips in uptrends and in bear
markets by selling rallies in downtrends.
"""

name = "12h_1d_ThreeBar_Pullback_TrendContinuation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d OHLCV for EMA34 Trend ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (20-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA34)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_12h[i]) or np.isnan(vol_ratio[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Three-bar pullback in uptrend: three consecutive lower closes
            pullback_down = (close[i-2] > close[i-1] > close[i])
            # Three-bar pullback in downtrend: three consecutive higher closes
            pullback_up = (close[i-2] < close[i-1] < close[i])
            
            # Long: uptrend (price above EMA34) + pullback down + volume
            if (close[i] > ema_34_12h[i] and 
                pullback_down and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price below EMA34) + pullback up + volume
            elif (close[i] < ema_34_12h[i] and 
                  pullback_up and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or three-bar continuation against position
            if position == 1:
                # Exit long: trend turns down OR three-bar pullback up against trend
                trend_down = close[i] < ema_34_12h[i]
                pullback_up = (close[i-2] < close[i-1] < close[i])
                if trend_down or pullback_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR three-bar pullback down against trend
                trend_up = close[i] > ema_34_12h[i]
                pullback_down = (close[i-2] > close[i-1] > close[i])
                if trend_up or pullback_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals