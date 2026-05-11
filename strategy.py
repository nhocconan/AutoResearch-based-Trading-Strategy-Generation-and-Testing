# Your final answer MUST be the full code for strategy.py only.
# All instructions from the system prompt apply.
# Write your strategy code below.

#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend: 50 EMA
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily pivot points from previous day (using daily data)
    # For 1d timeframe, we need to calculate pivots from previous day's OHLC
    # Since we are on 1d timeframe, we can use shift(1) to get previous day
    # But we need to access daily OHLC from the prices dataframe directly
    # We'll calculate pivots using the previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_point = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Weekly pivot levels (R1, S1, R2, S2)
    r1 = 2 * pivot_point - prev_low
    s1 = 2 * pivot_point - prev_high
    r2 = pivot_point + range_val
    s2 = pivot_point - range_val
    
    # Session filter: 00-24 UTC (whole day for daily timeframe)
    # For daily, we can use all hours, but we'll keep the concept
    # We'll use a simple filter: avoid extreme volatility days
    # Calculate 20-day ATR volatility regime
    atr_period = 20
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volatility regime: use ATR ratio to determine if we are in high/low vol
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr / atr_ma_50  # >1 = high volatility, <1 = low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1w trend or volatility data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_regime[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip first element due to roll
        if i == 0:
            continue
            
        # Volatility filter: only trade in normal to low volatility (avoid extreme volatility days)
        vol_filter = vol_regime[i] < 1.5  # Avoid extremely high volatility days
        
        if position == 0:
            # Long conditions: price breaks above R1 with 1w uptrend and volume confirmation
            if (high[i] > r1[i] and 
                close[i] > r1[i] and
                close[i] > ema50_1w_aligned[i] and  # 1w uptrend
                volume[i] > vol_ma_20[i] and       # volume spike
                vol_filter):                       # volatility filter
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 with 1w downtrend and volume confirmation
            elif (low[i] < s1[i] and 
                  close[i] < s1[i] and
                  close[i] < ema50_1w_aligned[i] and  # 1w downtrend
                  volume[i] > vol_ma_20[i] and        # volume spike
                  vol_filter):                        # volatility filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below S1 or reverses against trend
            if (low[i] < s1[i] or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above R1 or reverses against trend
            if (high[i] > r1[i] or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals