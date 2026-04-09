#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and ATR stoploss
# - Uses Williams %R(14) on 12h for oversold/overbought signals (long when %R < -80, short when %R > -20)
# - Requires 1d EMA(50) trend filter (long only when price > EMA50, short only when price < EMA50)
# - Uses ATR(14) for dynamic stoploss (2.5 * ATR) and position sizing (0.25)
# - Designed for range-bound markets with clear reversion to mean after extreme moves
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years) to avoid fee drag
# - Williams %R provides timely reversal signals; EMA50 filter avoids counter-trend trades

name = "12h_1d_williamsr_meanrev_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or mean reversion reversal
            if close[i] < prices['close'][i-1] - 2.5 * atr[i]:  # ATR stop
                position = 0
                signals[i] = 0.0
            elif williams_r[i] > -20:  # Overbought - exit long
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or mean reversion reversal
            if close[i] > prices['close'][i-1] + 2.5 * atr[i]:  # ATR stop
                position = 0
                signals[i] = 0.0
            elif williams_r[i] < -80:  # Oversold - exit short
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with trend filter
            if williams_r[i] < -80 and close[i] > ema_50_aligned[i]:  # Oversold + uptrend
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and close[i] < ema_50_aligned[i]:  # Overbought + downtrend
                position = -1
                signals[i] = -0.25
    
    return signals