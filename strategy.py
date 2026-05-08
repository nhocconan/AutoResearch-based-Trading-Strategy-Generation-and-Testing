#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_Volume_Trend_1d"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Keltner Channel on 4h: 20-period EMA, 2x ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    # Volume spike: current volume > 2.0 * 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma30 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Keltner with volume spike and 1d uptrend
            long_cond = (close[i] > upper[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below lower Keltner with volume spike and 1d downtrend
            short_cond = (close[i] < lower[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA20 (mean reversion)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner Channel breakout with volume confirmation and 1d EMA50 trend filter on 4h timeframe.
# Enters long when price breaks above upper Keltner band (20 EMA + 2*ATR) with volume spike and 1d uptrend.
# Enters short when price breaks below lower Keltner band with volume spike and 1d downtrend.
# Exits when price returns to the middle (EMA20), capturing mean reversion within the trend.
# Uses volatility-based channels that adapt to market conditions, effective in both trending and ranging markets.
# Targets 20-35 trades/year with strict volume confirmation (2.0x volume average) and trend filter.