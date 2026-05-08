#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA34 for trend
    close_4h = pd.Series(df_4h['close'].values)
    ema34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_4h = ema34_4h  # Use EMA34 as trend reference
    
    # Align 4h trend to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1h data for Camarilla pivot levels (using previous hour's OHLC)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla pivot levels calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 6)
    s1 = pivot - (range_val * 1.1 / 6)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(trend_4h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and 4h uptrend (price > EMA34)
            long_cond = (close[i] > r1[i] and vol_spike[i] and close[i] > trend_4h_aligned[i])
            
            # Short entry: price breaks below S1 with volume spike and 4h downtrend (price < EMA34)
            short_cond = (close[i] < s1[i] and vol_spike[i] and close[i] < trend_4h_aligned[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverses back above R1 (reversal signal)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation.
# Enters long when price breaks above R1 with volume spike and 4h uptrend (price > 4h EMA34).
# Enters short when price breaks below S1 with volume spike and 4h downtrend (price < 4h EMA34).
# Exits when price reverses back through S1/R1 respectively.
# Uses 4h trend for signal direction, 1h only for entry timing to reduce whipsaw.
# Session filter (08-20 UTC) avoids low-liquidity hours.
# Discrete sizing (0.20) minimizes churn. Target: 15-37 trades/year on 1h.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).