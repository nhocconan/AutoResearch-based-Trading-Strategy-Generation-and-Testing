#!/usr/bin/env python3
"""
1h Bollinger Band Width Breakout with 4h Trend Filter
Hypothesis: Low volatility contractions (BB width < 50th percentile) followed by breakouts in direction of 4h trend yield high-probability trades.
Works in both bull and bear markets by using volatility regime + trend alignment. Targets 15-37 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_bb_width_breakout_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 4h trend filter: EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data is NaN
        if (np.isnan(bb_width_percentile[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches middle band or volatility expands (width > 70th percentile)
            if (close[i] <= bb_middle[i] or 
                bb_width_percentile[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price touches middle band or volatility expands
            if (close[i] >= bb_middle[i] or 
                bb_width_percentile[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volatility contraction: BB width < 50th percentile (low volatility)
            vol_contract = bb_width_percentile[i] < 50
            
            # Trend direction
            uptrend = close[i] > ema_50_4h_aligned[i]
            downtrend = close[i] < ema_50_4h_aligned[i]
            
            # Breakout conditions
            breakout_up = close[i] > bb_upper[i]
            breakout_down = close[i] < bb_lower[i]
            
            # Long: upward breakout during low volatility + uptrend
            if (breakout_up and 
                vol_contract and 
                uptrend):
                position = 1
                signals[i] = 0.20
            # Short: downward breakout during low volatility + downtrend
            elif (breakout_down and 
                  vol_contract and 
                  downtrend):
                position = -1
                signals[i] = -0.20
    
    return signals