# 4h_HighLow_Pullback_Strategy_v1
# Hypothesis: Price tends to pull back to recent highs/lows during trends. Enter on pullbacks to 4h swing points
# in the direction of the 1-week trend. Use volume confirmation to avoid false breakouts.
# Works in bull/bear by following higher timeframe trend and using mean-reversion entries.
# Target: 20-40 trades/year per symbol with clear entry/exit rules.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_HighLow_Pullback_Strategy_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 50-period EMA on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 4-period EMA on 4h for entry timing
    ema_4_4h = pd.Series(close).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # Calculate rolling max/min for swing points (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_4_4h[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
            
        # Trend filter: 1-week EMA50 direction
        if i >= 51:
            trend_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            trend_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            trend_up = trend_down = False
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions
        if position == 0:
            # Long: pullback to recent low in uptrend
            if (trend_up and vol_confirm and 
                close[i] <= low_min_20[i] * 1.005 and  # Within 0.5% of 20-period low
                ema_4_4h[i] > ema_4_4h[i-1]):  # Short-term momentum turning up
                signals[i] = 0.25
                position = 1
            # Short: pullback to recent high in downtrend
            elif (trend_down and vol_confirm and 
                  close[i] >= high_max_20[i] * 0.995 and  # Within 0.5% of 20-period high
                  ema_4_4h[i] < ema_4_4h[i-1]):  # Short-term momentum turning down
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price reaches recent high or momentum fails
            if (close[i] >= high_max_20[i] * 0.995 or  # Near 20-period high
                ema_4_4h[i] < ema_4_4h[i-1]):  # Short-term momentum turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price reaches recent low or momentum fails
            if (close[i] <= low_min_20[i] * 1.005 or  # Near 20-period low
                ema_4_4h[i] > ema_4_4h[i-1]):  # Short-term momentum turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals