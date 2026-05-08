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
    
    # Get weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Daily data for pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(df_daily['high'].values, 1)
    prev_low = np.roll(df_daily['low'].values, 1)
    prev_close = np.roll(df_daily['close'].values, 1)
    prev_high[0] = df_daily['high'].values[0]
    prev_low[0] = df_daily['low'].values[0]
    prev_close[0] = df_daily['close'].values[0]
    
    # Camarilla pivot levels calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align Camarilla levels to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_daily, r1)
    s1_daily = align_htf_to_ltf(prices, df_daily, s1)
    
    # Weekly EMA34 for trend filter
    close_weekly = df_weekly['close'].values
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_daily = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or np.isnan(ema_34_daily[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and above weekly EMA34 (uptrend)
            long_cond = (close[i] > r1_daily[i] and vol_spike[i] and close[i] > ema_34_daily[i])
            
            # Short entry: price breaks below S1 with volume spike and below weekly EMA34 (downtrend)
            short_cond = (close[i] < s1_daily[i] and vol_spike[i] and close[i] < ema_34_daily[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > r1_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Camarilla R1/S1 breakout strategy with volume spike confirmation and weekly EMA34 trend filter.
# Enters long when price breaks above R1 with volume spike and price above weekly EMA34 (uptrend).
# Enters short when price breaks below S1 with volume spike and price below weekly EMA34 (downtrend).
# Exits when price reverses back through S1/R1 respectively.
# Uses discrete sizing (0.25) to minimize churn. Targets 15-30 trades/year on daily timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).