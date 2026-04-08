# 1d_1w_ema_trend_volume
# Hypothesis: On daily timeframe, use weekly EMA (10-period) as trend filter and price action for entry.
# Go long when price closes above weekly EMA with volume confirmation, short when price closes below weekly EMA with volume confirmation.
# Exit when price crosses back over weekly EMA or volume drops.
# This strategy aims for low trade frequency (<25/year) by requiring both trend alignment and volume spike.
# Works in bull/bear markets by only trading in direction of higher timeframe trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_trend_volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 10-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_10 = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    
    # Volume confirmation: volume > 2.0x 20-day average (high threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA not available
        if np.isnan(ema_10_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals with volume confirmation
        if not vol_confirm[i]:
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend change)
            if close[i] < ema_10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend change)
            if close[i] > ema_10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price closes above weekly EMA with volume confirmation
            if close[i] > ema_10_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below weekly EMA with volume confirmation
            elif close[i] < ema_10_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals