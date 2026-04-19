#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (34/89) with 4h trend filter and volume confirmation.
# Uses 4h EMA34 to define trend direction (bullish if close > EMA34, bearish if close < EMA34).
# Enters on 1h EMA34 crossing above/below EMA89 in direction of 4h trend, with volume > 1.5x 20-period average.
# Designed for 1h timeframe to capture medium-term trends while avoiding whipsaws via higher timeframe filter.
# Volume filter ensures momentum confirmation. Low trade frequency target: 15-30 trades/year.
name = "1h_EMA34_89_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA34 on 4h close
    df_4h = get_htf_data(prices, '4h')
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1h EMAs for entry timing
    ema34_1h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1h = pd.Series(close).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1h[i]) or np.isnan(ema89_1h[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema34_4h_aligned[i]
        trend_bearish = close[i] < ema34_4h_aligned[i]
        
        if position == 0:
            # Long: EMA34 crosses above EMA89, 4h trend bullish, volume filter
            if (ema34_1h[i] > ema89_1h[i] and ema34_1h[i-1] <= ema89_1h[i-1] and
                trend_bullish and volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: EMA34 crosses below EMA89, 4h trend bearish, volume filter
            elif (ema34_1h[i] < ema89_1h[i] and ema34_1h[i-1] >= ema89_1h[i-1] and
                  trend_bearish and volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if EMA34 crosses below EMA89 OR 4h trend turns bearish
            if (ema34_1h[i] < ema89_1h[i] and ema34_1h[i-1] >= ema89_1h[i-1]) or not trend_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if EMA34 crosses above EMA89 OR 4h trend turns bullish
            if (ema34_1h[i] > ema89_1h[i] and ema34_1h[i-1] <= ema89_1h[i-1]) or not trend_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals