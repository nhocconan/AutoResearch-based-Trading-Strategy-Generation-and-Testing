# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use 12h timeframe with daily Camarilla R1/S1 levels as breakout levels.
# Entry requires: price breaks R1/S1 + price above/below daily EMA34 (trend) + volume spike (current volume > 1.5x 20-period avg).
# Exit when price crosses back through the opposite S1/R1 level.
# Target: 12-37 trades/year on BTC/ETH. Uses daily trend filter to avoid counter-trend trades in choppy markets.
# Volume and trend filters reduce false breakouts. Works in bull (breakouts continue) and bear (breakouts fail quickly, exit fast).
# Risk managed via position exit on reversal, no separate stop needed due to timeframe.
# Discrete position size 0.25 to limit risk and reduce churn.

#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34) and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.083 * (high - low) * 1.1/2, S1 = close - 1.083 * (high - low) * 1.1/2
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.083 * range_1d * 1.1 / 2
    s1_1d = close_1d - 1.083 * range_1d * 1.1 / 2
    
    # Align 1d R1/S1 to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 12h volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data for EMA34 and volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (price > daily EMA34) + volume
            if close[i] > r1_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (price < daily EMA34) + volume
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite pivot level
            if position == 1:
                if close[i] < s1_1d_aligned[i]:  # Exit at S1 level
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_1d_aligned[i]:  # Exit at R1 level
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals