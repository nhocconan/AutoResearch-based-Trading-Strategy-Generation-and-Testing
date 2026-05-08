#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKeltnerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Weekly ATR for Keltner channels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_prev = np.roll(close_weekly, 1)
    close_weekly_prev[0] = close_weekly[0]
    tr = np.maximum(high_weekly - low_weekly,
                    np.maximum(np.abs(high_weekly - close_weekly_prev),
                               np.abs(low_weekly - close_weekly_prev)))
    tr[0] = high_weekly[0] - low_weekly[0]
    atr20_weekly = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr20_weekly)
    
    # Weekly Keltner channels
    upper_keltner = ema20_weekly_aligned + (2.0 * atr20_weekly_aligned)
    lower_keltner = ema20_weekly_aligned - (2.0 * atr20_weekly_aligned)
    
    # Daily volume filter: current volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(ema20_weekly_aligned[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Keltner channel with volume confirmation in uptrend
            if close[i] > upper_keltner[i] and close[i] > ema20_weekly_aligned[i] and volume[i] > vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Keltner channel with volume confirmation in downtrend
            elif close[i] < lower_keltner[i] and close[i] < ema20_weekly_aligned[i] and volume[i] > vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly EMA20 (trend reversal)
            if close[i] < ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above weekly EMA20 (trend reversal)
            if close[i] > ema20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Keltner breakout strategy with trend filter and volume confirmation.
# Uses weekly EMA20 and ATR to construct Keltner channels, entering long when price breaks above upper channel
# with volume confirmation in an uptrend (price > weekly EMA20), and short when price breaks below lower channel
# with volume confirmation in a downtrend (price < weekly EMA20). Exits when price crosses back below/above weekly EMA20.
# Designed to capture trends in both bull and bear markets while avoiding false breakouts with volume confirmation.
# Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag. Uses discrete sizing (0.25) to reduce churn.