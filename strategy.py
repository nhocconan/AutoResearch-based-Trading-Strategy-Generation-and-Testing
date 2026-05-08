#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Camarilla pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = np.roll(df_1w['high'].values, 1)
    prev_low = np.roll(df_1w['low'].values, 1)
    prev_close = np.roll(df_1w['close'].values, 1)
    prev_open = np.roll(df_1w['open'].values, 1)
    prev_high[0] = df_1w['high'].values[0]
    prev_low[0] = df_1w['low'].values[0]
    prev_close[0] = df_1w['close'].values[0]
    prev_open[0] = df_1w['open'].values[0]
    
    # Camarilla pivot levels calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align Camarilla levels to daily timeframe
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(ema_34_1d[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and above weekly EMA34 (uptrend)
            long_cond = (close[i] > r1_1d[i] and vol_spike[i] and close[i] > ema_34_1d[i])
            
            # Short entry: price breaks below S1 with volume spike and below weekly EMA34 (downtrend)
            short_cond = (close[i] < s1_1d[i] and vol_spike[i] and close[i] < ema_34_1d[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R1 (reversal signal)
            if close[i] > r1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout strategy with volume spike confirmation and weekly EMA34 trend filter on daily timeframe.
# Enters long when price breaks above R1 with volume spike and price above weekly EMA34 (uptrend).
# Enters short when price breaks below S1 with volume spike and price below weekly EMA34 (downtrend).
# Exits when price reverses back through S1/R1 respectively.
# Uses discrete sizing (0.25) to minimize churn. Targets 20-50 trades/year on daily timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended weekly levels).