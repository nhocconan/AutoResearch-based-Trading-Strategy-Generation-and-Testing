#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA34 for trend direction and daily volume confirmation.
# Enters long when price is above weekly EMA34 and daily volume is above average.
# Enters short when price is below weekly EMA34 and daily volume is above average.
# Uses weekly trend to capture major moves in both bull and bear markets.
# Volume filter reduces false signals and focuses on high conviction moves.
# Targets 10-25 trades/year (40-100 total over 4 years) with strict entry conditions.
# Works in bull/bear by following the higher timeframe trend.
name = "1d_1w_EMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA34 with volume confirmation
            if close[i] > ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34 with volume confirmation
            elif close[i] < ema_34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below weekly EMA34
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above weekly EMA34
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals