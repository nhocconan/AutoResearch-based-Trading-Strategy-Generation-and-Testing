# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Squeeze_v3
# Hypothesis: Combine Camarilla R1/S1 breakouts with 1d trend filter and volume squeeze detection to reduce false breakouts.
# Volume squeeze (low volatility breakout) captures explosive moves after consolidation, effective in both bull and bear markets.
# 12h timeframe targets 15-35 trades/year to minimize fee drag while capturing significant moves.
# Uses 1d EMA50 for trend, volume squeeze (current volume < 50% of 20-period average) as entry filter.
# Exit on opposite Camarilla level touch (R1 for longs, S1 for shorts) to capture mean reversion within the day.
# Position size 0.25 balances risk and return, with discrete levels to minimize churn.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Squeeze_v3"
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
    
    # Load 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d) / 3
    r1 = p + (high_1d - low_1d) * 1.1 / 12
    s1 = p - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume squeeze: current volume < 50% of 20-period average (low volatility breakout setup)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_squeeze = volume < (0.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA50 + volume squeeze (low volatility breakout)
            if close[i] > r1_aligned[i] and close[i] > ema_50_1d_aligned[i] and vol_squeeze[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below 1d EMA50 + volume squeeze (low volatility breakout)
            elif close[i] < s1_aligned[i] and close[i] < ema_50_1d_aligned[i] and vol_squeeze[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches or breaks below S1 (mean reversion within day)
            if close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches or breaks above R1 (mean reversion within day)
            if close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals