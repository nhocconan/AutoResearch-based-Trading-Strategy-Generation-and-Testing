# 1D_WeeklyTrend_Follow_With_Trend_Confirmation
# Hypothesis: Daily price follows weekly trend direction. Enter long when daily close crosses above weekly EMA20 and weekly EMA20 is rising, short when crosses below and falling.
# Uses weekly trend as filter to avoid counter-trend trades. Works in bull markets by catching uptrends and in bear markets by avoiding longs and taking shorts.
# Low trade frequency due to weekly trend filter requiring sustained direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly EMA20 for trend direction
    weekly_close = df_weekly['close'].values
    ema20_weekly = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema20_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: daily close crosses above weekly EMA20 and weekly EMA20 rising
            if close[i] > ema20_weekly_aligned[i] and close[i-1] <= ema20_weekly_aligned[i-1] and ema20_weekly_aligned[i] > ema20_weekly_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: daily close crosses below weekly EMA20 and weekly EMA20 falling
            elif close[i] < ema20_weekly_aligned[i] and close[i-1] >= ema20_weekly_aligned[i-1] and ema20_weekly_aligned[i] < ema20_weekly_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: daily close crosses back through weekly EMA20
            exit_signal = False
            if position == 1:
                # Exit long: close crosses below weekly EMA20
                if close[i] < ema20_weekly_aligned[i] and close[i-1] >= ema20_weekly_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: close crosses above weekly EMA20
                if close[i] > ema20_weekly_aligned[i] and close[i-1] <= ema20_weekly_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyTrend_Follow_With_Trend_Confirmation"
timeframe = "1d"
leverage = 1.0