# 4h_Williams_R13_Breakout_1wTrend_Volume
# Hypothesis: Williams %R (14) identifies overbought/oversold conditions. Combine with 1-week trend filter (EMA50) and volume spike for high-probability breakout trades. Works in bull (trend-following) and bear (mean reversion from extremes) markets. Target: 25-40 trades/year.
# Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long signal: Williams %R crosses above -80 from below (oversold bounce), price above 1w EMA50, volume spike
# Short signal: Williams %R crosses below -20 from above (overbought rejection), price below 1w EMA50, volume spike
# Exit: Williams %R crosses opposite threshold or price crosses 1w EMA50

#!/usr/bin/env python3
name = "4h_Williams_R13_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1w EMA to 4h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below, above 1w EMA50, volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                close[i] > ema50_1w_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, below 1w EMA50, volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  close[i] < ema50_1w_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -20 or price below 1w EMA50
            if williams_r[i] < -20 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -80 or price above 1w EMA50
            if williams_r[i] > -80 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals