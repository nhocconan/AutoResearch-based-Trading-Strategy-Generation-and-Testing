#!/usr/bin/env python3
"""
4h_1d_Camarilla_Reversal_MeanReversion
Hypothesis: Trade reversals at daily Camarilla H4/L4 levels during low volatility regimes.
Use Bollinger Band width percentile to identify range-bound markets (high probability of mean reversion).
Enter when price touches H4/L4 with rejection candle (close within 50% of range) and BBW < 30th percentile.
Exit on opposite level touch or BBW expansion above 70th percentile.
Designed for low-frequency mean reversion in ranging markets (works in both bull and bear markets as consolidation phases).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Reversal_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_1d = high_1d - low_1d
    
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === REGIME FILTER: BOLLINGER BAND WIDTH PERCENTILE ===
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean()
    std20 = close_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    bb_width = (upper_bb - lower_bb) / sma20
    
    # Percentile rank of BBW (lookback 50 periods)
    bbw_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Range regime: BBW below 30th percentile (low volatility)
    low_volatility = bbw_percentile < 30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(bbw_percentile[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Rejection candle: close within 50% of the day's range from the level
        daily_range = high_1d[-1] - low_1d[-1] if len(high_1d) > 0 else 0
        range_50pct = daily_range * 0.5
        
        # Long setup: price touches L4 and shows rejection (close above L4 + 50% range)
        long_setup = (low[i] <= l4_aligned[i] and 
                     close[i] > l4_aligned[i] + range_50pct and
                     low_volatility[i])
        
        # Short setup: price touches H4 and shows rejection (close below H4 - 50% range)
        short_setup = (high[i] >= h4_aligned[i] and 
                      close[i] < h4_aligned[i] - range_50pct and
                      low_volatility[i])
        
        # Exit: opposite level touch or volatility expansion (BBW > 70th percentile)
        volatility_expansion = bbw_percentile[i] > 70
        exit_long = (position == 1 and 
                    (high[i] >= h4_aligned[i] or volatility_expansion))
        exit_short = (position == -1 and 
                     (low[i] <= l4_aligned[i] or volatility_expansion))
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals