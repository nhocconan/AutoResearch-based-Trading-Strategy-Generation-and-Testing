#!/usr/bin/env python3
"""
1h Momentum Pullback with 4h Trend and Daily Volume Filter
Hypothesis: In strong 4h trends, pullbacks to EMA(21) on 1h with volume confirmation offer high-probability entries.
Works in bull/bear by only taking longs in 4h uptrends and shorts in 4h downtrends. Volume surge filters weak moves.
Target: 20-30 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_pullback_4h_trend_1d_volume_v1"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # EMA(21) on 1h for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # EMA(50) on 4h for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume ratio: current 1h volume vs 24-period average (approx 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma_24
    
    # Daily volume filter: today's volume > 1.5x 20-day average
    vol_ma_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    vol_filter = volume_1d > (vol_ma_20d * 1.5)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA(21) OR trend turns bearish
            if (close[i] < ema_21[i] or 
                close[i] <= ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA(21) OR trend turns bullish
            if (close[i] > ema_21[i] or 
                close[i] >= ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: 4h uptrend + pullback to EMA(21) + volume surge + daily volume confirmation
            if (close[i] > ema_50_4h_aligned[i] and  # 4h uptrend
                close[i] <= ema_21[i] * 1.005 and   # near or slightly below EMA(21) (pullback)
                vol_ratio[i] > 1.5 and              # 1h volume surge
                vol_filter_aligned[i]):             # daily volume confirmation
                position = 1
                signals[i] = 0.20
            # Short: 4h downtrend + bounce to EMA(21) + volume surge + daily volume confirmation
            elif (close[i] < ema_50_4h_aligned[i] and  # 4h downtrend
                  close[i] >= ema_21[i] * 0.995 and    # near or slightly above EMA(21) (bounce)
                  vol_ratio[i] > 1.5 and               # 1h volume surge
                  vol_filter_aligned[i]):              # daily volume confirmation
                position = -1
                signals[i] = -0.20
    
    return signals